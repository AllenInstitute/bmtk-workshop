import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import pandas as pd
import pickle as pkl
from math import pi
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "v1_model_utils"))
import other_v1_utils


# class StiffRegularizer(tf.keras.regularizers.Regularizer):
#     def __init__(self, strength, initial_value):
#         super().__init__()
#         self._strength = strength
#         self._initial_value = tf.Variable(initial_value, trainable=False)

#     def __call__(self, x):
#         return self._strength * tf.reduce_mean(tf.square(x - self._initial_value))
    
# class L2Regularizer(tf.keras.regularizers.Regularizer):
#     def __init__(self, strength, initial_value):
#         super().__init__()
#         self._strength = strength
#         self._initial_value = tf.Variable(initial_value, trainable=False)

#     def __call__(self, x):
#         return self._strength * tf.reduce_mean(tf.square(x))

class MeanStiffRegularizer(Layer):
    def __init__(self, strength, network, penalize_relative_change=False, dtype=tf.float32):
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype
        self._penalize_relative_change = penalize_relative_change
        # Compute voltage scale
        voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
        # Get the initial weights and properly scale them down
        indices = network["synapses"]["indices"]
        initial_value = np.array(network["synapses"]["weights"], dtype=np.float32)
        edge_type_ids = network['synapses']['edge_type_ids']
        # Scale initial values by the voltage scale of the node IDs
        initial_value /= voltage_scale[network['node_type_ids'][indices[:, 0]]]
        # Find unique values and their first occurrence indices
        unique_edge_types, self.idx = np.unique(edge_type_ids, return_inverse=True)
        # Sort first_occurrence_indices to maintain the order of first appearances
        self.num_unique = unique_edge_types.shape[0]
        sum_weights = np.bincount(self.idx, weights=initial_value, minlength=self.num_unique)
        count_weights = np.bincount(self.idx, minlength=self.num_unique)
        initial_mean_weights = sum_weights / count_weights
        # Determine target mean weights
        if self._penalize_relative_change:
            epsilon = np.float32(1e-4)
            denominator = np.maximum(np.abs(initial_mean_weights), epsilon)
            self._denominator = tf.constant(denominator, dtype=tf.float32)

        self.idx = tf.constant(self.idx, dtype=tf.int32)
        self.num_unique = tf.constant(self.num_unique, dtype=tf.int32)
        self._target_mean_weights = tf.constant(initial_mean_weights, dtype=tf.float32)

    @tf.function(jit_compile=True)
    def __call__(self, x):

        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        mean_edge_type_weights = tf.math.unsorted_segment_mean(x, self.idx, self.num_unique)
        if self._penalize_relative_change:
            # return self._strength * tf.reduce_mean(tf.abs(x - self._initial_value))
            relative_deviation = (mean_edge_type_weights - self._target_mean_weights) / self._denominator
            # Penalize the relative deviation
            reg_loss = tf.sqrt(tf.reduce_mean(tf.square(relative_deviation)))
        else:
            reg_loss = tf.reduce_mean(tf.square(mean_edge_type_weights - self._target_mean_weights))
        
        return tf.cast(reg_loss, dtype=self._dtype) * self._strength
    
class MeanStdStiffRegularizer(Layer):
    def __init__(self, strength, network, penalize_relative_change=False, 
                    std_weight=0.5, logspace=True, dtype=tf.float32):
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype
        self._penalize_relative_change = penalize_relative_change
        self._std_weight = std_weight  # Weight for std deviation component
        self._logspace = logspace  # Whether to use logspace for std calculation
        self._epsilon = tf.constant(1e-6, dtype=dtype)  # Small value to avoid log(0)
        
        # Compute voltage scale
        voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
        # Get the initial weights and properly scale them down
        indices = network["synapses"]["indices"]
        initial_value = np.array(network["synapses"]["weights"], dtype=np.float32)
        edge_type_ids = network['synapses']['edge_type_ids']
        # Scale initial values by the voltage scale of the node IDs
        initial_value /= voltage_scale[network['node_type_ids'][indices[:, 0]]]
        # Find unique values and their first occurrence indices
        unique_edge_types, self.idx = np.unique(edge_type_ids, return_inverse=True)
        # Sort first_occurrence_indices to maintain the order of first appearances
        self.num_unique = unique_edge_types.shape[0]            
        sum_weights = np.bincount(self.idx, weights=initial_value, minlength=self.num_unique)
        count_weights = np.bincount(self.idx, minlength=self.num_unique)
        initial_mean_weights = sum_weights / count_weights
        
        # Calculate std deviation per edge type in logspace if requested
        self._target_std_weights = []
        for i in range(self.num_unique):
            edge_type_weights = initial_value[self.idx == i]
            if self._logspace:
                # Use abs to handle any potential negative weights
                log_weights = np.log(np.abs(edge_type_weights) + np.float32(1e-6))
                std = np.std(log_weights)
            else:
                std = np.std(edge_type_weights)
            self._target_std_weights.append(std)
        self._target_std_weights = tf.constant(self._target_std_weights, dtype=self._dtype)
        
        # Determine target mean weights and denominators for relative change
        if self._penalize_relative_change:
            epsilon = np.float32(1e-4)
            denominator = np.maximum(np.abs(initial_mean_weights), epsilon)
            self._denominator = tf.constant(denominator, dtype=tf.float32)
            # Also for std deviation
            epsilon = np.float32(1e-3)
            std_denominator = np.maximum(np.abs(self._target_std_weights.numpy()), epsilon)
            self._std_denominator = tf.constant(std_denominator, dtype=tf.float32)

        self.idx = tf.constant(self.idx, dtype=tf.int32)
        self.num_unique = tf.constant(self.num_unique, dtype=tf.int32)
        self._target_mean_weights = tf.constant(initial_mean_weights, dtype=tf.float32)

    @tf.function(jit_compile=True)
    def __call__(self, x):

        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        # Calculate mean per edge type
        mean_edge_type_weights = tf.math.unsorted_segment_mean(x, self.idx, self.num_unique)

        # Calculate std deviation per edge type
        if self._logspace:
            # Use abs for log transformation to handle potential negative weights
            abs_x = tf.abs(x)
            log_x = tf.math.log(abs_x + self._epsilon)
            mean_log_x = tf.math.unsorted_segment_mean(log_x, self.idx, self.num_unique)
            # Get log values for each input position
            gathered_mean_log_x = tf.gather(mean_log_x, self.idx)
            # Calculate squared differences
            squared_diffs = tf.square(log_x - gathered_mean_log_x)
            # Calculate variance and std dev with epsilon for stability
            log_var = tf.math.unsorted_segment_mean(squared_diffs, self.idx, self.num_unique)
            std_edge_type_weights = tf.sqrt(log_var + self._epsilon)        # prevent division by zero for edge_types with just 1 edge
        else:
            # Calculate variance directly with improved stability
            gathered_means = tf.gather(mean_edge_type_weights, self.idx)
            squared_diffs = tf.square(x - gathered_means)
            var_edge_type_weights = tf.math.unsorted_segment_mean(squared_diffs, self.idx, self.num_unique)
            std_edge_type_weights = tf.sqrt(var_edge_type_weights + self._epsilon) # prevent division by zero for edge_types with just 1 edge

        # Calculate losses with improved numerical stability
        if self._penalize_relative_change:
            # Mean deviation component - with safe division
            mean_relative_deviation = (mean_edge_type_weights - self._target_mean_weights) / self._denominator
            mean_loss = tf.sqrt(tf.reduce_mean(tf.square(mean_relative_deviation)))
            
            # Std deviation component - with safe division
            std_relative_deviation = (std_edge_type_weights - self._target_std_weights) / self._std_denominator
            std_loss = tf.sqrt(tf.reduce_mean(tf.square(std_relative_deviation)))
        else:
            # Mean deviation component
            mean_squared_error = tf.square(mean_edge_type_weights - self._target_mean_weights)
            mean_loss = tf.reduce_mean(mean_squared_error)
            
            # Std deviation component
            std_squared_error = tf.square(std_edge_type_weights - self._target_std_weights)
            std_loss = tf.reduce_mean(std_squared_error)
        
        # Combine losses with weighting
        total_loss = (1.0 - self._std_weight) * mean_loss + self._std_weight * std_loss
        return tf.cast(total_loss, dtype=self._dtype) * self._strength
                
                    
class StiffKLLogNormalRegularizer(Layer):
    """Regularization using KL divergence for log-normal distributions

    Args:
        Layer (_type_): _description_
    """
    def __init__(self, strength, network, dtype=tf.float32):
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype
        self.epsilon = tf.constant(1e-8, dtype=dtype)

        # Compute voltage scale and rescale initial weights
        voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
        indices = network["synapses"]["indices"]
        initial_value = np.array(network["synapses"]["weights"], dtype=np.float32)
        edge_type_ids = network['synapses']['edge_type_ids']
        initial_value /= voltage_scale[network['node_type_ids'][indices[:, 0]]]

        # Edge type indexing
        unique_edge_types, self.idx = np.unique(edge_type_ids, return_inverse=True)
        self.idx = tf.constant(self.idx, dtype=tf.int32)
        self.num_unique = tf.constant(unique_edge_types.shape[0], dtype=tf.int32)
        # Filter edge types with more than 2 connections
        count_weights = np.bincount(self.idx, minlength=self.num_unique)
        self.edges_mask = count_weights > 2
        # Create indices for valid edge types (for gather operations)
        self.valid_indices = tf.convert_to_tensor(np.where(self.edges_mask)[0], dtype=tf.int32)
        self.num_valid = tf.shape(self.valid_indices)[0]

        # Precompute mean and std of log(initial weights) per edge type
        log_initial_value = np.log(np.abs(initial_value) + 1e-10)  # Add small epsilon in numpy
        self._target_log_mean_all = tf.math.unsorted_segment_mean(
            tf.constant(log_initial_value, dtype), self.idx, self.num_unique
        )
        # Variance per edge type
        log_squared_diff = (log_initial_value - self._target_log_mean_all.numpy()[self.idx]) ** 2
        log_var = tf.math.unsorted_segment_mean(
            tf.constant(log_squared_diff, dtype), self.idx, self.num_unique
        )
        log_std_all = tf.sqrt(log_var + self.epsilon)  # Add epsilon before sqrt

        # Pre-filter target values using gather instead of boolean_mask
        self._target_log_mean = tf.gather(self._target_log_mean_all, self.valid_indices)
        self._target_log_std = tf.gather(log_std_all, self.valid_indices)

    @tf.function(jit_compile=True)
    def __call__(self, x):
        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        # Calculate log of absolute values with epsilon for stability
        log_x = tf.math.log(tf.abs(x) + self.epsilon)
        # Calculate mean and std of log(x) per edge type
        log_mean_all = tf.math.unsorted_segment_mean(log_x, self.idx, self.num_unique)
        # Calculate std deviation with better numerical stability
        squared_diff = tf.square(log_x - tf.gather(log_mean_all, self.idx))
        log_var_all = tf.math.unsorted_segment_mean(squared_diff, self.idx, self.num_unique)
        log_std_all = tf.sqrt(log_var_all + self.epsilon)  # Add epsilon before sqrt
        # Use gather instead of boolean_mask
        log_mean = tf.gather(log_mean_all, self.valid_indices)
        log_std = tf.gather(log_std_all, self.valid_indices)
        # KL divergence calculation with improved numerical stability
        log_ratio = tf.math.log(self._target_log_std + self.epsilon) - tf.math.log(log_std + self.epsilon)
        denominator = 2.0 * tf.square(self._target_log_std) + self.epsilon
        std_ratio = tf.square(log_std) / denominator
        diff_ratio = tf.square(log_mean - self._target_log_mean) / denominator
        
        # Combine terms after stable calculations
        kl = log_ratio + std_ratio + diff_ratio - 0.5
        # Use reduce_mean without abs since KL should be positive
        kl_mean = tf.reduce_mean(kl)

        return self._strength * tf.cast(kl_mean, dtype=self._dtype)

class L2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, network, flags, penalize_relative_change=False, dtype=tf.float32):
        super().__init__()
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype

        if penalize_relative_change:
            # Compute voltage scale
            voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
            # Get the initial weights and properly scale them down
            indices = network["synapses"]["indices"]
            weights = np.array(network["synapses"]["weights"], dtype=np.float32)
            # Scale initial values by the voltage scale of the node IDs
            voltage_scale_node_ids = voltage_scale[network['node_type_ids'][indices[:, 0]]]
            initial_value = weights / voltage_scale_node_ids
            # using the edge_type ids group calculate the mean weight of each type of edge in the network and then create a constant with same shape as weights and with each value corresponding to the populations mean
            # Calculate mean weights for each edge type
            edge_type_ids = np.array(network['synapses']['edge_type_ids'])
            unique_edge_type_ids, inverse_indices = np.unique(edge_type_ids, return_inverse=True)
            mean_weights = np.array([np.mean(initial_value[edge_type_ids == edge_type_id]) for edge_type_id in unique_edge_type_ids])
            # Create target mean weights array based on the edge type indices
            self._target_mean_weights = tf.constant(mean_weights[inverse_indices], dtype=tf.float32)
            epsilon = tf.constant(1e-4, dtype=tf.float32)  # A small constant to avoid division by zero
            self._target_mean_weights = tf.maximum(tf.abs(self._target_mean_weights), epsilon)
        else:
            self._target_mean_weights = None

    @tf.function(jit_compile=True)
    def __call__(self, x):
        
        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        if self._target_mean_weights is None:
            return tf.cast(self._strength * tf.reduce_mean(tf.square(x)), dtype=self._dtype)
        else:
            relative_deviation = x / self._target_mean_weights
            mse = self._strength * tf.reduce_mean(tf.square(relative_deviation))
            return tf.cast(mse, dtype=self._dtype)


def spike_trimming(spikes, pre_delay=50, post_delay=50, trim=True):
    pre = pre_delay or 0
    if trim:
        post = -post_delay if post_delay else None
        spikes = spikes[:, pre:post, :]
    else:
        spikes = spikes[:, pre:, :]
    return spikes

def sample_firing_rates(firing_rates, n_neurons, rnd_seed):
    # Sort the original firing rates
    sorted_firing_rates = np.sort(firing_rates)
    # Calculate the empirical cumulative distribution function (CDF)
    percentiles = np.linspace(0, 1, sorted_firing_rates.size)
    # Generate random uniform values from 0 to 1
    rate_rd = np.random.RandomState(seed=rnd_seed)
    x_rand = rate_rd.uniform(low=0, high=1, size=n_neurons)
    # Use inverse transform sampling: interpolate the uniform values to find the firing rates
    target_firing_rates = np.sort(np.interp(x_rand, percentiles, sorted_firing_rates))
    # target_firing_rates = np.interp(x_rand, percentiles, sorted_firing_rates)    
    return target_firing_rates

def huber_quantile_loss(u, tau, kappa, dtype=tf.float32):
    tau = tf.cast(tau, dtype)
    abs_u = tf.abs(u)
    num = tf.abs(tau - tf.cast(u <= 0, dtype))
    branch_1 = num / (2 * kappa) * tf.square(u)
    branch_2 = num * (abs_u - 0.5 * kappa)
    return tf.where(abs_u <= kappa, branch_1, branch_2)

### To calculate the loss of firing rates between neuron types
def compute_spike_rate_target_loss(rates, target_rates, dtype=tf.float32):
    # TODO: define this function
    # target_rates is a dictionary that contains all the cell types.
    # I should iterate on them, and add the cost for each one at the end.
    # spikes will have a shape of (batch_size, n_steps, n_neurons)
    # rates = tf.reduce_mean(_spikes, (0, 1))
    total_loss = tf.constant(0.0, dtype=dtype)
    num_neurons = tf.constant(0, dtype=tf.int32)
    # if core_mask is not None:
    #     core_neurons_ids = np.where(core_mask)[0]

    for key, value in target_rates.items():
        neuron_ids = value["neuron_ids"]
        if len(neuron_ids) != 0:
            _rate_type = tf.gather(rates, neuron_ids)
            target_rate = value["sorted_target_rates"]
            # if core_mask is not None:
            #     key_core_mask = np.isin(value["neuron_ids"], core_neurons_ids)
            #     neuron_ids =  np.where(key_core_mask)[0]
            #     _rate_type = tf.gather(rates, neuron_ids)
            #     target_rate = value["sorted_target_rates"][key_core_mask]
            # else:
            #     _rate_type = tf.gather(rates, value["neuron_ids"])
            #     target_rate = value["sorted_target_rates"]

            loss_type = compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=dtype)
            total_loss += tf.reduce_sum(loss_type)
            num_neurons += tf.size(neuron_ids)
        
    total_loss /= tf.cast(num_neurons, dtype=dtype)  

    return total_loss

def compute_spike_rate_distribution_loss(_rates, target_rate, dtype=tf.float32):
    # Firstly we shuffle the current model rates to avoid bias towards a particular tuning angles (inherited from neurons ordering in the network)
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rates = tf.gather(_rates, rand_ind)
    sorted_rate = tf.sort(_rates)
    # u = target_rate - sorted_rate
    u = sorted_rate - target_rate
    # tau = (tf.range(target_rate.shape[0]), dtype) + 1) / target_rate.shape[0]
    tau = (tf.range(target_rate.shape[0]) + 1) / target_rate.shape[0]
    loss = huber_quantile_loss(u, tau, 0.002, dtype=dtype)
    # loss = huber_quantile_loss(u, tau, 0.1, dtype=dtype)

    return loss

def process_neuropixels_data(path=''):
    # Load data
    neuropixels_data_path = f'Neuropixels_data/cortical_metrics_1.4.csv'
    df_all = pd.read_csv(neuropixels_data_path, sep=",")
    # Exc and PV have sufficient number of cells, so we'll filter out non-V1 Exc and PV.
    # SST and VIP are small populations, so let's keep also non-V1 neurons
    exclude = (df_all["cell_type"].isnull() | df_all["cell_type"].str.contains("EXC") | df_all["cell_type"].str.contains("PV")) \
            & (df_all["ecephys_structure_acronym"] != 'VISp')
    df = df_all[~exclude]
    print(f"Original: {df_all.shape[0]} cells,   filtered: {df.shape[0]} cells")

    # Some cells have very large values of RF. They are likely not-good fits, so ignore.
    df.loc[(df["width_rf"] > 100), "width_rf"] = np.nan
    df.loc[(df["height_rf"] > 100), "height_rf"] = np.nan

    # Save the processed table
    df.to_csv(f'Neuropixels_data/v1_OSI_DSI_DF.csv', sep=" ", index=False)
    # return df

def neuropixels_cell_type_to_cell_type(pop_name):
    if not isinstance(pop_name, str):
        return pop_name
    if ' ' in pop_name:  # This is already new. No need to update.
        return pop_name

    # Convert pop_name in the neuropixels cell type to cell types. E.g, 'EXC_L23' -> 'L2/3 Exc', 'PV_L5' -> 'L5 PV'
    layer = pop_name.split('_')[1]
    class_name = pop_name.split('_')[0]
    if "2" in layer:
        layer = "L2/3"
    elif layer == "L1":
        return "L1 Htr3a"  # special case
    if class_name == "EXC":
        class_name = "Exc"
    if class_name == 'Htr3a':
        class_name = 'VIP'

    return f"{layer} {class_name}"


class SpikeRateDistributionTarget:
    """ Instead of regularization, treat it as a target.
        The main difference is that this class will calculate the loss
        for each subtypes of the neurons."""
    def __init__(self, network, spontaneous_fr=False, rate_cost=.5, pre_delay=None, post_delay=None, 
                 data_dir='GLIF_network', core_mask=None, rates_dampening=1.0, seed=42, dtype=tf.float32,
                 neuropixels_df='Neuropixels_data/v1_OSI_DSI_DF.csv'):
        self._network = network
        self._rate_cost = rate_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._rates_dampening = rates_dampening
        self._core_mask = core_mask
        if self._core_mask is not None:
            self._np_core_mask = self._core_mask.numpy()
        self._data_dir = data_dir
        self._dtype = dtype
        self._seed = seed
        self._neuropixels_df = neuropixels_df
        if spontaneous_fr:
            self.neuropixels_feature = 'firing_rate_sp'
        else:
            self.neuropixels_feature = 'Ave_Rate(Hz)'
        self._target_rates = self.get_neuropixels_firing_rates()

    def get_neuropixels_firing_rates(self):
        """
        Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load data
        # neuropixels_data_path = f'Neuropixels_data/v1_OSI_DSI_DF.csv'

        neuropixels_data_path = self._neuropixels_df
        if neuropixels_data_path == 'Neuropixels_data/v1_OSI_DSI_DF.csv':
            if not os.path.exists(neuropixels_data_path):
                process_neuropixels_data(path=neuropixels_data_path)
        else: # just inform the user that the custom file is loading.
            print(f"Using custom neuropixels data file for FR loss: {neuropixels_data_path}")

        # New dataset has Spont_Rate(Hz) instead of firing_rate_sp.
        # if reading firing_rate_sp fails, replace it with Spont_Rate(Hz) and try again.
        features_to_load = ['ecephys_unit_id', 'cell_type', 'firing_rate_sp', 'Ave_Rate(Hz)']
        try:
            np_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')
        except ValueError:
            print(f"Neuropixels data file {neuropixels_data_path} does not contain firing_rate_sp. Using Spont_Rate(Hz) instead.")
            features_to_load = ['ecephys_unit_id', 'cell_type', 'Spont_Rate(Hz)', 'Ave_Rate(Hz)']
            np_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')
            # Rename the column to match the original
            np_df.rename(columns={'Spont_Rate(Hz)': 'firing_rate_sp'}, inplace=True)
        area_node_types = pd.read_csv(os.path.join(self._data_dir, f'network/v1_node_types.csv'), sep=" ")
        # Ensure they use the new names
        np_df["cell_type"] = np_df["cell_type"].apply(neuropixels_cell_type_to_cell_type)

        # Define population queries
        query_mapping = {
            "i1H": 'L1 Htr3a',
            "e23": 'L2/3 Exc',
            "i23P": 'L2/3 PV',
            "i23S": 'L2/3 SST',
            "i23V": 'L2/3 VIP',
            "e4": 'L4 Exc',
            "i4P": 'L4 PV',
            "i4S": 'L4 SST',
            "i4V": 'L4 VIP',
            "e5": 'L5 Exc',
            "i5P": 'L5 PV',
            "i5S": 'L5 SST',
            "i5V": 'L5 VIP',
            "e6": 'L6 Exc',
            "i6P": 'L6 PV',
            "i6S": 'L6 SST',
            "i6V": 'L6 VIP'
        }

        # Define the reverse mapping
        reversed_query_mapping = {v:k for k, v in query_mapping.items()}
        # Process rates
        type_rates_dict = {
                            reversed_query_mapping[cell_type]: np.append(subdf[self.neuropixels_feature].dropna().values / 1000, 0)
                            # reversed_query_mapping[cell_type]: np.sort(np.append(subdf[self.neuropixels_feature].dropna().values / 1000, 0)) # the rates are sorted again later so is redundant
                            for cell_type, subdf in np_df.groupby("cell_type")
                        }
        # Identify node_type_ids for each population query
        pop_ids = {query: area_node_types[area_node_types.pop_name.str.contains(query)].index.values for query in query_mapping.keys()}
        # Create a dictionary with rates and IDs
        target_firing_rates = {pop_query: {'rates': type_rates_dict[pop_query], 'ids': pop_ids[pop_query]} for pop_query in pop_ids.keys()}
        
        for key, value in target_firing_rates.items():
            # identify tne ids that are included in value["ids"]
            neuron_ids = np.where(np.isin(self._network["node_type_ids"], value["ids"]))[0]
            if self._core_mask is not None:
                # if core_mask is not None, use only neurons in the core
                neuron_ids = neuron_ids[self._np_core_mask[neuron_ids]]

            type_n_neurons = len(neuron_ids)
            target_firing_rates[key]['neuron_ids'] = tf.convert_to_tensor(neuron_ids, dtype=tf.int32)
            sorted_target_rates = self._rates_dampening * sample_firing_rates(value["rates"], type_n_neurons, self._seed)
            target_firing_rates[key]['sorted_target_rates'] = tf.convert_to_tensor(sorted_target_rates, dtype=self._dtype)

        return target_firing_rates    

    def __call__(self, spikes, trim=True):
        # if trim:
        #     if self._pre_delay is not None:
        #         spikes = spikes[:, self._pre_delay:, :]
        #     if self._post_delay is not None and self._post_delay != 0:
        #         spikes = spikes[:, :-self._post_delay, :]

        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)
        rates = tf.reduce_mean(spikes, (0, 1)) # calculate the mean firing rate over time and batch

        reg_loss = compute_spike_rate_target_loss(rates, self._target_rates, dtype=self._dtype) 

        return reg_loss * self._rate_cost

# class SpikeRateDistributionRegularization:
#     def __init__(self, target_rates, rate_cost=0.5, dtype=tf.float32):
#         self._rate_cost = rate_cost
#         self._target_rates = target_rates
#         self._dtype = dtype

#     def __call__(self, spikes):
#         reg_loss = (
#             compute_spike_rate_distribution_loss(spikes, self._target_rates, dtype=self._dtype)
#             * self._rate_cost
#         )
#         reg_loss = tf.reduce_sum(reg_loss)

#         return reg_loss

class SynchronizationLoss(Layer):
    def __init__(self, network, sync_cost=10., t_start=0., t_end=0.5, n_samples=50, data_dir='Synchronization_data', 
                 session='evoked', dtype=tf.float32, core_mask=None, seed=42, **kwargs):
        super(SynchronizationLoss, self).__init__(dtype=dtype, **kwargs)
        self._sync_cost = sync_cost
        self._t_start = t_start
        self._t_end = t_end
        self._t_start_seconds = int(t_start * 1000)
        self._t_end_seconds = int(t_end * 1000)
        self._core_mask = core_mask
        self._data_dir = data_dir
        self._dtype = dtype
        self._n_samples = n_samples
        self._base_seed = seed

        pop_names = other_v1_utils.pop_names(network)
        if self._core_mask is not None:
            pop_names = pop_names[core_mask]
        node_ei = np.array([pop_name[0] for pop_name in pop_names])
        node_id = np.arange(len(node_ei))
        # Get the IDs for excitatory neurons
        node_id_e = node_id[node_ei == 'e']
        self.node_id_e = tf.constant(node_id_e, dtype=tf.int32) # 14423
        # Pre-define bin sizes (same as experimental data)
        bin_sizes = np.logspace(-3, 0, 20)
        # using the simulation length, limit bin_sizes to define at least 2 bins
        bin_sizes_mask = bin_sizes < (self._t_end - self._t_start)/2
        self.bin_sizes = bin_sizes[bin_sizes_mask]
        self.epsilon = 1e-7  # Small constant to avoid division by zero

        # Load the experimental data
        duration = str(int((t_end - t_start) * 1000))
        experimental_data_path = os.path.join(data_dir, f'Fano_factor_v1', f'v1_fano_running_{duration}ms_{session}.npy')
        # experimental_data_path = os.path.join(data_dir, f'all_fano_300ms_{session}.npy')
        assert os.path.exists(experimental_data_path), f'File not found: {experimental_data_path}'
        experimental_fanos = np.load(experimental_data_path, allow_pickle=True)
        experimental_fanos_mean = np.nanmean(experimental_fanos[:, bin_sizes_mask], axis=0)
        self.experimental_fanos_mean = tf.constant(experimental_fanos_mean, dtype=self._dtype)

    def pop_fano_tf(self, spikes, bin_sizes):
        spikes = tf.expand_dims(spikes, axis=-1)
        fanos = tf.TensorArray(dtype=self._dtype, size=len(bin_sizes))
        for i, bin_width in enumerate(bin_sizes):
            bin_size = int(np.round(bin_width * 1000))
            # Use convolution for efficient binning
            kernel = tf.ones((bin_size, 1, 1), dtype=self._dtype)
            convolved = tf.nn.conv1d(spikes, kernel, stride=bin_size, padding="VALID")
            sp_counts = tf.squeeze(convolved, axis=-1)  # Shape: (60, new_width)
            # Compute mean and variance of spike counts
            mean_count = tf.reduce_mean(sp_counts, axis=1)
            var_count = tf.math.reduce_variance(sp_counts, axis=1)
            mean_count = tf.maximum(mean_count, self.epsilon)
            # fanos.append(tf.reduce_mean(var_count / mean_count))
            fano_per_sample = var_count / mean_count  # => [n_samples]
            fano = tf.reduce_mean(fano_per_sample)
            fanos = fanos.write(i, fano)

        return fanos.stack()


    def __call__(self, spikes, trim=True):

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)

        if trim:
            spikes = spikes[:, self._t_start_seconds:self._t_end_seconds, :]
            bin_sizes = self.bin_sizes
            experimental_fanos_mean = self.experimental_fanos_mean
        else:
            t_start = 0
            t_end = spikes.shape[1] / 1000
            # using the simulation length, limit bin_sizes to define at least 2 bins
            bin_sizes_mask = self.bin_sizes < (t_end - t_start)/2
            bin_sizes = self.bin_sizes[bin_sizes_mask]
            experimental_fanos_mean = self.experimental_fanos_mean[bin_sizes_mask]
        
        spikes = tf.cast(spikes, self._dtype)  
        # choose random trials to sample from (usually we only have 1 trial to sample from)
        n_trials = tf.shape(spikes)[0]
        # increase the base seed to avoid the same random neurons to be selected in every instantiation of the class
        self._base_seed += 1
        sample_trials = tf.random.uniform([self._n_samples], minval=0, maxval=n_trials, dtype=tf.int32, seed=self._base_seed)
        # Generate sample counts with a normal distribution
        sample_size = 70
        sample_std = 30
        sample_counts = tf.cast(tf.random.normal([self._n_samples], mean=sample_size, stddev=sample_std, seed=self._base_seed), tf.int32)
        sample_counts = tf.clip_by_value(sample_counts, clip_value_min=15, clip_value_max=tf.shape(self.node_id_e)[0]) # clip the values to be between 15 and 14423
        # Randomize the neuron ids
        shuffled_e_ids = tf.random.shuffle(self.node_id_e, seed=self._base_seed)
        selected_spikes_sample = tf.TensorArray(self._dtype, size=self._n_samples)
        previous_id = tf.constant(0, dtype=tf.int32)
        for i in tf.range(self._n_samples):
            sample_num = sample_counts[i] # 40 #68
            sample_trial = sample_trials[i] # 0
            ## randomly choose sample_num ids from self.node_id_e with replacement
            ## sample_ids = tf.random.shuffle(self.node_id_e)[:sample_num]
            ## randomly choose sample_num ids from shuffled_ids without replacement
            if previous_id + sample_num > tf.size(shuffled_e_ids):
                # shuffled_e_ids = tf.random.shuffle(self.node_id_e, seed=self._base_seed)
                shuffled_e_ids = tf.random.shuffle(shuffled_e_ids, seed=self._base_seed)
                previous_id = tf.constant(0, dtype=tf.int32)
            sample_ids = shuffled_e_ids[previous_id:previous_id+sample_num]
            previous_id += sample_num
            
            selected_spikes = tf.reduce_sum(tf.gather(spikes[sample_trial], sample_ids, axis=1), axis=-1)
            selected_spikes_sample = selected_spikes_sample.write(i, selected_spikes)

        selected_spikes_sample = selected_spikes_sample.stack()
        fanos_mean = self.pop_fano_tf(selected_spikes_sample, bin_sizes=bin_sizes)
        # # Calculate MSE between the experimental and calculated Fano factors
        mse_loss = tf.reduce_mean(tf.square(experimental_fanos_mean - fanos_mean))
        # # Calculate the synchronization loss
        sync_loss = self._sync_cost * mse_loss

        return sync_loss
   

class VoltageRegularization:
    def __init__(self, cell, voltage_cost=1e-5, dtype=tf.float32, core_mask=None):
        self._voltage_cost = voltage_cost
        self._cell = cell
        self._dtype = dtype
        self._core_mask = core_mask
        # self._voltage_offset = tf.cast(self._cell.voltage_offset, dtype)
        # self._voltage_scale = tf.cast(self._cell.voltage_scale, dtype)
        # if core_mask is not None:
        #     self._voltage_offset = tf.boolean_mask(self._voltage_offset, core_mask)
        #     self._voltage_scale = tf.boolean_mask(self._voltage_scale, core_mask)

    def __call__(self, voltages):
        if self._core_mask is not None:
            voltages = tf.boolean_mask(voltages, self._core_mask, axis=2)
            
        # voltages = (voltages - self._voltage_offset) / self._voltage_scale
        # v_pos = tf.square(tf.nn.relu(voltages - 1.0))
        # v_neg = tf.square(tf.nn.relu(-voltages + 1.0))
        # voltage_loss = tf.reduce_mean(tf.reduce_mean(v_pos + v_neg, -1))
        v_tot = tf.square(voltages - 1.0)
        voltage_loss = tf.reduce_mean(v_tot)

        return voltage_loss * self._voltage_cost
    

class CustomMeanLayer(Layer):
    def call(self, inputs):
        spike_rates, mask = inputs
        masked_data = tf.boolean_mask(spike_rates, mask)
        return tf.reduce_mean(masked_data)


class OrientationSelectivityLoss:
    def __init__(self, network=None, osi_cost=1e-5, pre_delay=None, post_delay=None, dtype=tf.float32, 
                 core_mask=None, method="crowd_osi", subtraction_ratio=1.0, layer_info=None,
                 neuropixels_df="Neuropixels_data/v1_OSI_DSI_DF.csv"):
        
        self._network = network
        self._osi_cost = osi_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._dtype = dtype
        self._core_mask = core_mask
        self._method = method
        self._subtraction_ratio = subtraction_ratio  # only for crowd_spikes method
        self._tf_pi = tf.constant(np.pi, dtype=dtype)
        self._neuropixels_df = neuropixels_df
        if (self._core_mask is not None) and (self._method == "crowd_spikes" or self._method == "crowd_osi"):
            self.np_core_mask = self._core_mask.numpy()
            core_tuning_angles = network['tuning_angle'][self.np_core_mask]
            self._tuning_angles = tf.constant(core_tuning_angles, dtype=dtype)
        else:
            self._tuning_angles = tf.constant(network['tuning_angle'], dtype=dtype)
        
        if self._method == "neuropixels_fr":
            self._layer_info = layer_info  # needed for neuropixels_fr method
            # the layer_info should be a dictionary that contains
            # the cell id of the corresponding layer.
            # the keys should be something like "EXC_L23" or "PV_L5"   

        elif self._method == "crowd_osi":
            # Get the target OSI
            self._target_osi_dsi = self.get_neuropixels_osi_dsi()
            self._min_rates_threshold = tf.constant(0.0005, dtype=self._dtype)
            # sum the core_mask
            n_nodes = len(self._tuning_angles)
            # self.node_type_ids = tf.zeros(n_nodes, dtype=tf.int32)
            node_type_ids = np.zeros(n_nodes, dtype=np.int32)
            osi_target_values = []
            dsi_target_values = []
            cell_type_count = []
            for node_type_id, (key, value) in enumerate(self._target_osi_dsi.items()):
                node_ids = value['ids']
                osi_target_values.append(value['OSI'])
                dsi_target_values.append(value['DSI'])
                cell_type_count.append(len(node_ids))
                # update the ndoe_type_ids tensor in positions node_ids with the node_type_id
                # self.node_type_ids = tf.tensor_scatter_nd_update(self.node_type_ids, indices=tf.expand_dims(node_ids, axis=1), updates=tf.fill(tf.shape(node_ids), node_type_id))
                node_type_ids[node_ids] = node_type_id

            self.osi_target_values = tf.constant(osi_target_values, dtype=self._dtype)
            self.dsi_target_values = tf.constant(dsi_target_values, dtype=self._dtype)
            self.cell_type_count = tf.constant(cell_type_count, dtype=self._dtype)
            self.node_type_ids = tf.constant(node_type_ids, dtype=tf.int32)
            self._n_node_types = len(self._target_osi_dsi)

    def calculate_delta_angle(self, stim_angle, tuning_angle):
        # angle unit is degrees.
        # this function calculates the difference between stim_angle and tuning_angle,
        # but it is fine to have the opposite direction.
        # so, delta angle is always between -90 and 90.
        # they are both vector, so dimension matche is needed.
        # stim_angle is a length of batch size
        # tuning_angle is a length of n_neurons

        # delta_angle = stim_angle - tuning_angle
        delta_angle = tf.expand_dims(stim_angle, axis=1) - tuning_angle
        delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
        delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)
        # # do it twice to make sure everything is between -90 and 90.
        delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
        delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)

        return delta_angle
    
    def get_neuropixels_osi_dsi(self):
        """
        Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load data
        # neuropixels_data_path = f'Neuropixels_data/v1_OSI_DSI_DF.csv'
        neuropixels_data_path = self._neuropixels_df
        # if the default one is specified and the file doesn't exist, process the data
        if neuropixels_data_path == "Neuropixels_data/v1_OSI_DSI_DF.csv":
            if not os.path.exists(neuropixels_data_path):
                process_neuropixels_data(path=neuropixels_data_path)
        else:
            print(f"Using custom neuropixels data file for OSI/DSI loss: {neuropixels_data_path}")
        features_to_load = ['ecephys_unit_id', 'cell_type', 'OSI', 'DSI', "Ave_Rate(Hz)", "max_mean_rate(Hz)"]
        osi_dsi_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')
        
        nonresponding = osi_dsi_df["max_mean_rate(Hz)"] < 0.5
        osi_dsi_df.loc[nonresponding, "OSI"] = np.nan
        osi_dsi_df.loc[nonresponding, "DSI"] = np.nan
        osi_dsi_df = osi_dsi_df[osi_dsi_df["Ave_Rate(Hz)"] != 0]
        osi_dsi_df.dropna(inplace=True)
        osi_dsi_df["cell_type"] = osi_dsi_df["cell_type"].apply(neuropixels_cell_type_to_cell_type)
        osi_target = osi_dsi_df.groupby("cell_type")['OSI'].mean()
        dsi_target = osi_dsi_df.groupby("cell_type")['DSI'].mean()

        original_pop_names = other_v1_utils.pop_names(self._network)
        if self._core_mask is not None:
            original_pop_names = original_pop_names[self.np_core_mask] 

        cell_types = np.array([other_v1_utils.pop_name_to_cell_type(pop_name, ignore_l5e_subtypes=True) for pop_name in original_pop_names])
        node_ids = np.arange(len(cell_types))
        cell_ids = {key: node_ids[cell_types == key] for key in set(osi_dsi_df['cell_type'])}

        # osi_target = osi_df.groupby("cell_type")['OSI'].mean()
        # osi_target = osi_df.groupby("cell_type")['OSI'].median()
        # osi_df.groupby("cell_type")['OSI'].median()
        # convert to dict
        osi_dsi_exp_dict = {key: {'OSI': val, 'DSI': dsi_target[key], 'ids': cell_ids[key]} for key, val in osi_target.to_dict().items()}

        return osi_dsi_exp_dict
        
    def vonmises_model_fr(self, structure, population):
        from scipy.stats import vonmises
        paramdic = self._von_mises_params
        _params = paramdic[structure][population]
        if len(_params) == 4:
            mu, kappa, a, b = _params
        vonmises_pdf = vonmises(kappa, loc=mu).pdf
        
        angles = np.deg2rad(np.arange(-85, 86, 10)) * 2  # *2 needed to make it proper model
        model_fr = a + b * vonmises_pdf(angles)

        return model_fr
    
    def neuropixels_fr_loss(self, spikes, angle):
        # if the trget fr is not set, construct them
        if not hasattr(self, "_target_frs"):

            # self._von_mises_params = np.load("GLIF_network/param_dict_orientation.npy")
            # pickle instead
            with open("GLIF_network/param_dict_orientation.pkl", 'rb') as f:
                self._von_mises_params = pkl.load(f)
            # get the model values with 10 degree increments 
            structure = "VISp"
            self._target_frs = {}
            for key in self._layer_info.keys():
                self._target_frs[key] = self.vonmises_model_fr(structure, key)
                # TODO: convert it to tensor if needed.
        
        # assuming 1 ms bins
        spike_rates = tf.reduce_mean(spikes, axis=[0, 1]) / spikes.shape[1] * 1000
        angle_bins = tf.constant(np.arange(-90, 91, 10), dtype=self._dtype)
        nbins = angle_bins.shape[0] - 1
        # now, process each layer
        # losses = tf.TensorArray(tf.float32, size=len(self._layer_info))
        losses = []
        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        
        custom_mean_layer = CustomMeanLayer()

        
        for key, value in self._layer_info.items():
            # first, calculate delta_angle
            
            # rates = tf.TensorArray(tf.float32, size=nbins)
            rates_list = []
            for i in range(nbins):
                mask = (delta_angle >= angle_bins[i]) & (delta_angle < angle_bins[i+1])
                # take the intersection with core mask
                mask = tf.logical_and(mask, self._core_mask)
                mask = tf.logical_and(mask, value)
                # mask = mask.flatten()
                # doesn't work.
                mask = tf.reshape(mask, [-1])
                mean_val = custom_mean_layer([spike_rates, mask])
                # rates_ = rates.write(i, mean_val)
                rates_list.append(mean_val)
                # rates = rates.write(i, tf.reduce_mean(tf.boolean_mask(spike_rates, mask)))

            # calculate the loss
            # rates = rates.stack()
            rates = tf.stack(rates_list)
            loss = tf.reduce_mean(tf.square(rates - self._target_frs[key]))
            # if key == "EXC_L6":
                # print the results!
                # tf.print("Layer6: ", rates)
                # tf.print("target: ", self._target_frs[key])
            # losses = losses.write(i, loss)
            losses.append(loss)
        
        # final_loss = tf.reduce_sum(losses.stack()) * self._osi_cost
        final_loss = tf.reduce_mean(tf.stack(losses)) * self._osi_cost
        return final_loss
        
    def crowd_spikes_loss(self, spikes, angle):
        # I need to access the tuning angle. of all the neurons.
        angle = tf.cast(angle, self._dtype)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
            
        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        # sum spikes in _z, and multiply with delta_angle.
        mean_spikes = tf.reduce_mean(spikes, axis=[1]) 
        mean_angle = mean_spikes * delta_angle
        # Here, the expected value with random firing to subtract
        # (this prevents the osi loss to drive the firing rates to go to zero.)
        expected_sum_angle = tf.reduce_mean(mean_spikes) * 45
        
        angle_loss = tf.reduce_mean(tf.abs(mean_angle)) - expected_sum_angle * self._subtraction_ratio
        
        return angle_loss * self._osi_cost
    
    def crowd_osi_loss(self, spikes, angle, normalizer=None):  
        # Ensure angle is [batch_size] and cast to correct dtype
        angle = tf.cast(tf.reshape(angle, [-1]), self._dtype)  # [batch_size]
        # Compute delta_angle with broadcasting
        delta_angle = angle[:, tf.newaxis] - self._tuning_angles[tf.newaxis, :]  # [batch_size, n_neurons_core]
        radians_delta_angle = delta_angle * (self._tf_pi / 180)
            
        # Compute rates over time dimension
        rates = tf.reduce_mean(spikes, axis=1)  # [batch_size, n_neurons]
        if self._core_mask is not None:
            rates = tf.boolean_mask(rates, self._core_mask, axis=1)

        if normalizer is not None:
            if self._core_mask is not None:
                normalizer = tf.boolean_mask(normalizer, self._core_mask, axis=0)
            # Use tf.maximum to ensure each element of normalizer does not fall below min_normalizer_value
            normalizer = tf.maximum(normalizer, self._min_rates_threshold)
            rates = rates / normalizer

        # Instead of complex numbers, use cosine and sine separately
        weighted_osi_cos_responses = rates * tf.math.cos(2.0 * radians_delta_angle)
        weighted_dsi_cos_responses = rates * tf.math.cos(radians_delta_angle)

        batch_size = tf.shape(rates)[0]
        # Adjust segment_ids for batch dimension
        batch_offsets = tf.range(batch_size, dtype=self.node_type_ids.dtype) * self._n_node_types  # [batch_size]
        batch_offsets_expanded = batch_offsets[:, tf.newaxis]  # [batch_size, 1]

        segment_ids = self.node_type_ids[tf.newaxis, :]  # [1, n_neurons_core]
        segment_ids = tf.tile(segment_ids, [batch_size, 1])  # [batch_size, n_neurons_core]
        segment_ids = segment_ids + batch_offsets_expanded  # [batch_size, n_neurons_core]

        # Flatten data and segment_ids
        data_flat_rates = tf.reshape(rates, [-1])  # [batch_size * n_neurons_core]
        data_flat_weighted_osi = tf.reshape(weighted_osi_cos_responses, [-1])
        data_flat_weighted_dsi = tf.reshape(weighted_dsi_cos_responses, [-1])
        segment_ids_flat = tf.reshape(segment_ids, [-1])

        num_segments = batch_size * self._n_node_types

        # Compute denominators and numerators
        approximated_denominator = tf.math.unsorted_segment_mean(data_flat_rates, segment_ids_flat, num_segments=num_segments)
        approximated_denominator = tf.reshape(approximated_denominator, [batch_size, self._n_node_types])
        approximated_denominator = tf.maximum(approximated_denominator, self._min_rates_threshold)

        osi_numerator = tf.math.unsorted_segment_mean(data_flat_weighted_osi, segment_ids_flat, num_segments=num_segments)
        osi_numerator = tf.reshape(osi_numerator, [batch_size, self._n_node_types])

        dsi_numerator = tf.math.unsorted_segment_mean(data_flat_weighted_dsi, segment_ids_flat, num_segments=num_segments)
        dsi_numerator = tf.reshape(dsi_numerator, [batch_size, self._n_node_types])

        # Compute approximations
        osi_approx_type = osi_numerator / approximated_denominator  # [batch_size, n_node_types]
        dsi_approx_type = dsi_numerator / approximated_denominator

        # Average over batch size
        osi_approx_type = tf.reduce_mean(osi_approx_type, axis=0)
        dsi_approx_type = tf.reduce_mean(dsi_approx_type, axis=0)

        # Compute losses
        # osi_target_values = self.osi_target_values[tf.newaxis, :]  # [1, n_node_types]
        # dsi_target_values = self.dsi_target_values[tf.newaxis, :]  # [1, n_node_types]
        osi_loss_type = tf.math.square(osi_approx_type - self.osi_target_values)  # [n_node_types]
        dsi_loss_type = tf.math.square(dsi_approx_type - self.dsi_target_values)
    
        # cell_type_count = self.cell_type_count[tf.newaxis, :]  # [1, n_node_types]
        numerator = tf.reduce_sum((osi_loss_type + dsi_loss_type) * self.cell_type_count)  # [1]
        denominator = tf.reduce_sum(self.cell_type_count)  # Scalar

        # total_loss_per_batch = numerator / denominator  # [batch_size]
        # total_loss = tf.reduce_mean(total_loss_per_batch) * self._osi_cost

        total_loss = (numerator / denominator) * self._osi_cost

        return total_loss

    def __call__(self, spikes, angle, trim, normalizer=None):

        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)

        if self._method == "crowd_osi":
            return self.crowd_osi_loss(spikes, angle, normalizer=normalizer)
        elif self._method == "crowd_spikes":
            return self.crowd_spikes_loss(spikes, angle)
        elif self._method == "neuropixels_fr":
            return self.neuropixels_fr_loss(spikes, angle)


class EarthMoversDistanceRegularizer(Layer):
    """
    Regularizer that penalizes the Earth Mover's Distance (Wasserstein-1) between the current and initial
    synaptic weight distributions, per edge type, averaged over all edge types.
    Uses TF operations for initialization and tf.map_fn in call for memory efficiency.
    """
    def __init__(self, strength, network, dtype=tf.float32):
        super().__init__()
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype

        # --- Original Initialization Logic ---
        voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
        indices = network["synapses"]["indices"]
        initial_value_np = np.array(network["synapses"]["weights"], dtype=np.float32)
        # edge_type_ids_np = network['synapses']['edge_type_ids']
        # use the connection_type_ids instead
        edge_type_ids_np = other_v1_utils.connection_type_ids(network)
        initial_value_np /= voltage_scale[network['node_type_ids'][indices[:, 0]]]
        n_edges = len(initial_value_np)
        # --- End Original Initialization Logic ---

        # Convert to TF Tensors
        initial_value = tf.constant(initial_value_np, dtype=tf.float32)
        edge_type_ids = tf.constant(edge_type_ids_np, dtype=tf.int32)
        
        unique_edge_types, idx = np.unique(edge_type_ids, return_inverse=True)
        self.num_unique = tf.constant(unique_edge_types.shape[0], dtype=tf.int32)

        self._initial_value = tf.constant(initial_value, dtype=tf.float32)
        
        
        # presort the initial value
        for i in tf.range(self.num_unique):
            mask = tf.equal(idx, i)
            y_i = tf.boolean_mask(self._initial_value, mask)
            y_i = tf.sort(y_i)
            self._initial_value = tf.tensor_scatter_nd_update(self._initial_value, tf.where(mask), y_i)


        ### 2. Reorder original_indices and initial_value based on sorted edge types
        original_indices = tf.range(n_edges, dtype=tf.int32)
        sorted_indices = tf.argsort(edge_type_ids)
        permuted_original_indices = tf.gather(original_indices, sorted_indices)
        sorted_edge_type_ids = tf.gather(edge_type_ids, sorted_indices) # Needed for unique

        # 3. Find unique edge types and row indices for ragged tensor construction
        unique_types, row_indices = tf.unique(sorted_edge_type_ids)
        nrows = tf.shape(unique_types)[0]

        # 4. Construct group_indices RaggedTensor (indices into the *original* weight tensor)
        self._group_indices = tf.RaggedTensor.from_value_rowids(
            values=permuted_original_indices,
            value_rowids=row_indices,
            nrows=nrows
        )

    @tf.function(jit_compile=False) # Do not use jit_compile=True. It uses a lot of memory.
    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)
        emd_losses = tf.TensorArray(self._dtype, size=self.num_unique)
        for i in tf.range(self.num_unique):
            x_i = tf.gather(x, self._group_indices[i])
            y_i = tf.gather(self._initial_value, self._group_indices[i])

            # y_i is presorted.
            emd = tf.reduce_mean(tf.abs(tf.sort(x_i) - y_i))
            emd_losses = emd_losses.write(i, emd)
        emd_losses = emd_losses.stack()
        reg_loss = tf.reduce_mean(emd_losses)
        return reg_loss * self._strength
