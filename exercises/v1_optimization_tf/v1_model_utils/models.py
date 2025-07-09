import numpy as np
import tensorflow as tf
import os 
import pickle as pkl
# import subprocess
from . import other_v1_utils
# from cuda_ops.synaptic_currents.synaptic_currents_ops import calculate_synaptic_currents_cuda

# Define a custom gradient for the spike function.
# Diverse functions can be used to define the gradient.
# Here we provide variations of this functions depending on
# the gradient type and the precision of the input tensor.

def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude

def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)

# def slayer_pseudo(v_scaled, sigma, amplitude):
#     return tf.math.exp(-sigma * tf.abs(v_scaled)) * amplitude

# @tf.custom_gradient
# def spike_slayer(v_scaled, sigma, amplitude):
#     z_ = tf.greater(v_scaled, 0.0)
#     z_ = tf.cast(z_, tf.float32)

#     def grad(dy):
#         de_dz = dy
#         dz_dv_scaled = slayer_pseudo(v_scaled, sigma, amplitude)

#         de_dv_scaled = de_dz * dz_dv_scaled

#         return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

#     return tf.identity(z_, name="spike_slayer"), grad

@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None, None]

    return tf.identity(z_, name="spike_gauss"), grad

@tf.custom_gradient
def spike_gauss_16(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None, None]

    return tf.identity(z_, name="spike_gauss"), grad


@tf.custom_gradient
def spike_gauss_b16(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.bfloat16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None, None]

    return tf.identity(z_, name='spike_gauss'), grad

@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None]

    return tf.identity(z_, name="spike_function"), grad

@tf.custom_gradient
def spike_function_16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None]

    return tf.identity(z_, name="spike_function"), grad

@tf.custom_gradient
def spike_function_b16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.bfloat16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def calculate_synaptic_currents(rec_z_buf, synapse_indices, weight_values, dense_shape, 
                                synaptic_basis_weights, syn_ids, pre_ind_table):
    # Get the batch size and number of neurons
    batch_size = tf.cast(tf.shape(rec_z_buf)[0], dtype=tf.int64)
    n_post_neurons = tf.cast(dense_shape[0], dtype=tf.int64)
    n_syn_basis = tf.cast(tf.shape(synaptic_basis_weights)[1], dtype=tf.int64)  # Number of receptor types
    # Find the indices of non-zero inputs in x_t
    # non_zero_indices: [num_non_zero_spikes, 2], columns are [batch_index, pre_neuron_index]
    non_zero_indices = tf.where(rec_z_buf > 0)
    batch_indices = non_zero_indices[:, 0]         
    pre_neuron_indices = non_zero_indices[:, 1]
    # Retrieve relevant connections and weights for these pre_neurons
    new_indices, new_weights, new_syn_ids, post_in_degree, all_synaptic_inds = get_new_inds_table(
        synapse_indices, weight_values, syn_ids, pre_neuron_indices, pre_ind_table
    )
    # Repeat batch_indices to match total_num_connections
    batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
    post_neuron_indices = new_indices[:, 0]
    # We will sum over all connections to get currents for each neuron and each basis
    num_segments = batch_size * n_post_neurons
    segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
    # Gather the factor sets for the synapses involved
    new_weights = tf.expand_dims(new_weights, axis=1)  # [total_num_connections, 1]
    new_weights = new_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
    # if per_type_training:
    #     per_type_weights = tf.expand_dims(tf.gather(recurrent_per_type_weight_values, 
    #                                                 tf.gather(connection_type_ids, all_synaptic_inds)), axis=1)
    #     new_weights = new_weights * per_type_weights
    # Calculate the total recurrent current received by each neuron per basis dimension
    i_rec_flat = tf.math.unsorted_segment_sum(
        new_weights,
        segment_ids,
        num_segments=num_segments
    )

    def grad(dy, variables=None):
        # Reshape gradient dy to match batch x neuron dimensions
        # dy_reshaped = tf.reshape(dy, [batch_size, n_post_neurons, n_syn_basis])
        # Define a function to process each receptor type
        def per_receptor_grad(r_id):
            # Extract the gradient for this receptor type (shape: [batch_size, n_post_neurons])
            dy_r = tf.reshape(dy[:, r_id], [batch_size, n_post_neurons])
            # dy_r = dy_reshaped[:, :, r_id]
            # Compute gradient w.r.t rec_z_buf for this receptor type
            recurrent_weights_factors = tf.gather(synaptic_basis_weights[:, r_id], syn_ids, axis=0)
            weights_syn_receptors = weight_values * recurrent_weights_factors
            sparse_w_rec = tf.sparse.SparseTensor(synapse_indices, weights_syn_receptors, dense_shape)
            de_dv_rid = tf.sparse.sparse_dense_matmul(dy_r, sparse_w_rec, adjoint_a=False)
            
            return de_dv_rid  # shape: [batch_size, n_pre_neurons]
        # Use tf.map_fn to apply per_receptor_grad to each receptor index
        # tf.map_fn will return a tensor of shape [n_syn_basis, batch_size, n_pre_neurons]
        de_dv_all = tf.map_fn(per_receptor_grad, 
                            tf.range(n_syn_basis), 
                            dtype=dy.dtype,
                            parallel_iterations=1)
        # Sum over all receptors
        # de_dv_all: [n_syn_basis, batch_size, n_pre_neurons]
        de_dv = tf.reduce_sum(de_dv_all, axis=0)  # [batch_size, n_pre_neurons]
        de_dv = tf.cast(de_dv, dtype=rec_z_buf.dtype)

        # # Extract the gradient for this receptor type (shape: [batch_size, n_post_neurons])
        # r_id = 0
        # dy_r = tf.reshape(dy[:, r_id], [batch_size, n_post_neurons])
        # # dy_r = dy_reshaped[:, :, r_id]
        # # Compute gradient w.r.t rec_z_buf for this receptor type
        # recurrent_weights_factors = tf.gather(synaptic_basis_weights[:, r_id], syn_ids, axis=0)
        # weights_syn_receptors = weight_values * recurrent_weights_factors
        # sparse_w_rec = tf.sparse.SparseTensor(synapse_indices, weights_syn_receptors, dense_shape)
        # de_dv_rid = tf.sparse.sparse_dense_matmul(dy_r, sparse_w_rec, adjoint_a=False)
        # de_dv = tf.cast(de_dv_rid, dtype=rec_z_buf.dtype)
            
        # Gradient w.r.t weight_values
        dnew_weights = tf.gather(dy, segment_ids)  # Match connections
        dbasis_scaled_grad = dnew_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
        de_dweight_values_connection = tf.reduce_sum(dbasis_scaled_grad, axis=1)       
        # Instead of tensor_scatter_nd_add, use unsorted_segment_sum:
        de_dweight_values = tf.math.unsorted_segment_sum(
            data=de_dweight_values_connection,
            segment_ids=all_synaptic_inds,
            num_segments=tf.shape(weight_values)[0]
        )

        return [
            de_dv,              # Gradient w.r.t rec_z_buf
            None,               # synapse_indices (constant)
            de_dweight_values,    # Gradient w.r.t weight_values
            None,                 # dense_shape[0] (constant)
            None,                 # dense_shape[1] (constant)
            None,                 # synaptic_basis_weights (constant)
            None,                 # syn_ids (constant)
            None                  # pre_ind_table (constant)
        ]

    return i_rec_flat, grad

def exp_convolve(tensor, decay=0.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], 0
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse,
                       initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered

def make_pre_ind_table(indices, n_source_neurons=197613):
    """
    This function creates a table that maps presynaptic indices to 
    the indices of the recurrent_indices tensor using a RaggedTensor.
    This approach ensures that every presynaptic neuron, even those with no
    postsynaptic connections, has an entry in the RaggedTensor.
    """
    # Extract presynaptic IDs
    pre_ids = indices[:, 1]  # shape: [num_synapses]
    # Sort the synapses by presynaptic ID
    sort_idx = tf.argsort(pre_ids, axis=0)
    sorted_pre = tf.gather(pre_ids, sort_idx)
    # Count how many synapses belong to each presynaptic neuron
    # (We cast to int32 for tf.math.bincount.)
    counts = tf.math.bincount(tf.cast(sorted_pre, tf.int32), minlength=n_source_neurons)
    # Build row_splits to define a RaggedTensor from these sorted indices
    row_splits = tf.concat([[0], tf.cumsum(counts)], axis=0)
    # The values of the RaggedTensor are the original synapse-array row indices,
    # but sorted by presyn neuron
    rt = tf.RaggedTensor.from_row_splits(sort_idx, row_splits, validate=False)

    return rt

def get_new_inds_table(indices, weights, syn_ids, non_zero_cols, pre_ind_table):
    """Optimized function that prepares new sparse indices tensor."""
    # Gather the rows corresponding to the non_zero_cols
    selected_rows = tf.gather(pre_ind_table, non_zero_cols)
    # Flatten the selected rows to get all_inds
    all_synapse_inds = selected_rows.flat_values
    # get the number of postsynaptic neurons 
    post_in_degree = selected_rows.row_lengths()
    # Gather from indices, weights and syn_ids using all_inds
    new_indices = tf.gather(indices, all_synapse_inds)
    new_weights = tf.gather(weights, all_synapse_inds)
    new_syn_ids = tf.gather(syn_ids, all_synapse_inds)

    return new_indices, new_weights, new_syn_ids, post_in_degree, all_synapse_inds

class SignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, positive):
        self._positive = positive

    def __call__(self, w):
        condition = tf.greater(self._positive, 0)  # yields bool
        sign_corrected_w = tf.where(condition, tf.nn.relu(w), -tf.nn.relu(-w))
        return sign_corrected_w


class SparseSignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, mask, positive):
        self._mask = mask
        self._positive = positive

    def __call__(self, w):
        condition = tf.greater(self._positive, 0)  # yields bool
        sign_corrected_w = tf.where(condition, tf.nn.relu(w), -tf.nn.relu(-w))
        return tf.where(self._mask, sign_corrected_w, tf.zeros_like(sign_corrected_w))


class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, lower_limit, upper_limit):
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit

    def __call__(self, w):
        return tf.clip_by_value(w, self._lower_limit, self._upper_limit)


class V1Column(tf.keras.layers.Layer):
    def __init__(
        self,
        network,
        lgn_input,
        bkg_input,
        dt=1.0,
        gauss_std=0.5,
        dampening_factor=0.3,
        recurrent_dampening_factor=0.5,
        input_weight_scale=1.0,
        recurrent_weight_scale=1.0,
        lr_scale=1.0,
        spike_gradient=False,
        max_delay=5,
        batch_size=1,
        bkg_firing_rate=250,
        pseudo_gauss=False,
        train_recurrent=True,
        train_recurrent_per_type=True,
        train_input=False,
        train_noise=True,
        hard_reset=False,
        current_input=False
    ):
        super().__init__()
        _params = dict(network["node_params"])
        # Rescale the voltages to have them near 0, as we wanted the effective step size
        # for the weights to be normalized when learning (weights are scaled similarly)
        voltage_scale = _params["V_th"] - _params["E_L"]
        voltage_offset = _params["E_L"]
        _params["V_th"] = (_params["V_th"] - voltage_offset) / voltage_scale
        _params["E_L"] = (_params["E_L"] - voltage_offset) / voltage_scale
        _params["V_reset"] = (_params["V_reset"] - voltage_offset) / voltage_scale
        _params["asc_amps"] = (_params["asc_amps"] / voltage_scale[..., None])  # _params['asc_amps'] has shape (111, 2)
        # Define the other model variables
        self._node_type_ids = np.array(network["node_type_ids"])
        self._dt = tf.constant(dt, self.compute_dtype)
        self._recurrent_dampening = tf.constant(recurrent_dampening_factor, self.compute_dtype)
        self._dampening_factor = tf.constant(dampening_factor, self.compute_dtype)
        self._pseudo_gauss = pseudo_gauss
        self._lr_scale = tf.constant(lr_scale, dtype=self.compute_dtype)
        self._spike_gradient = spike_gradient
        self._hard_reset = hard_reset
        self._current_input = current_input
        self._n_neurons = int(network["n_nodes"])
        self._bkg_firing_rate = bkg_firing_rate
        self._gauss_std = tf.constant(gauss_std, self.compute_dtype)
        # Determine the membrane time decay constant
        tau = _params["C_m"] / _params["g"]
        membrane_decay = np.exp(-dt / tau)
        current_factor = 1 / _params["C_m"] * (1 - membrane_decay) * tau

        # Determine the synaptic dynamic parameters for each of the 5 basis receptors.
        path='synaptic_data/tau_basis.npy' # [0.7579732  1.33243834 2.34228851 4.11750046 7.23813909]
        tau_syns = np.load(path)
        self._n_syn_basis = tau_syns.size
        syn_decay = np.exp(-dt / tau_syns)
        syn_decay = tf.constant(syn_decay, dtype=self.compute_dtype)
        syn_decay = tf.tile(syn_decay, [self._n_neurons])
        self.syn_decay = tf.expand_dims(syn_decay, axis=0) # expand the dimension for processing different receptor types
        psc_initial = np.e / tau_syns
        psc_initial = tf.constant(psc_initial, dtype=self.compute_dtype)
        psc_initial = tf.tile(psc_initial, [self._n_neurons])
        self.psc_initial = tf.expand_dims(psc_initial, axis=0) # expand the dimension for processing different receptor types

        # Find the maximum delay in the network
        self.max_delay = int(np.round(np.min([np.max(network["synapses"]["delays"]), max_delay])))
        self.batch_size = batch_size
        
        def _gather(prop):
            return tf.gather(prop, self._node_type_ids)
    
        def _f(_v, trainable=False):
            return tf.Variable(tf.cast(_gather(_v), self.compute_dtype), trainable=trainable)

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        def custom_val(_v, trainable=False):
            _v = tf.Variable(
                tf.cast(inv_sigmoid(_gather(_v)), self.compute_dtype),
                trainable=trainable,
            )
            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        # Gather the neuron parameters for every neuron
        self.t_ref = _f(_params["t_ref"])  # refractory time
        self.v_reset = _f(_params["V_reset"])
        self.asc_amps = _f(_params["asc_amps"], trainable=False)
        _k = tf.cast(_params['k'], self.compute_dtype)
        # inverse sigmoid of the adaptation rate constant (1/ms)
        param_k, param_k_read = custom_val(_k, trainable=False)
        k = param_k_read()
        self.exp_dt_k = tf.exp(-self._dt * k)
        self.v_th = _f(_params["V_th"])
        self.v_gap = self.v_reset - self.v_th
        e_l = _f(_params["E_L"])
        self.normalizer = self.v_th - e_l
        param_g = _f(_params["g"])
        self.gathered_g = param_g * e_l
        self.decay = _f(membrane_decay)
        self.current_factor = _f(current_factor)
        self.voltage_scale = _f(voltage_scale)
        self.voltage_offset = _f(voltage_offset)

        # Find the synaptic basis representation for each synaptic type
        # path = os.path.join('GLIF_network', 'syn_id_to_syn_weights_dict.pkl')
        path = os.path.join(network["data_dir"], 'tf_data', 'syn_id_to_syn_weights_dict.pkl')
        with open(path, "rb") as f:
            syn_id_to_syn_weights_dict = pkl.load(f)
        synaptic_basis_weights = np.array(list(syn_id_to_syn_weights_dict.values()))
        self.synaptic_basis_weights = tf.constant(synaptic_basis_weights, dtype=self.variable_dtype)

        ### Network recurrent connectivity ###
        indices = np.array(network["synapses"]["indices"])
        weights = np.array(network["synapses"]["weights"])
        dense_shape = np.array(network["synapses"]["dense_shape"])
        syn_ids = np.array(network["synapses"]["syn_ids"])
        delays = np.array(network["synapses"]["delays"])
        # Scale down the recurrent weights
        weights = (weights/voltage_scale[self._node_type_ids[indices[:, 0]]])      
        # Use the maximum delay to clip the synaptic delays
        delays = np.round(np.clip(delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the presynaptic neuron indices
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)
        self.recurrent_dense_shape = dense_shape[0], self.max_delay * dense_shape[1] 
        #the first column (presynaptic neuron) has size n_neurons and the second column (postsynaptic neuron) has size max_delay*n_neurons
        # Define the Tensorflow variables
        self.recurrent_indices = tf.Variable(indices, dtype=tf.int64, trainable=False)
        self.pre_ind_table = make_pre_ind_table(indices, n_source_neurons=self.recurrent_dense_shape[1])
        # add dimension for the weights factors - TensorShape([23525415, 1])
        # weights = tf.expand_dims(weights, axis=1) 
        # Set the sign of the connections (exc or inh)
        # recurrent_weight_positive = tf.Variable(
        #     weights >= 0.0, name="recurrent_weights_sign", trainable=False)
        recurrent_weight_positive = tf.constant(weights >= 0, dtype=tf.int8)

        # if training the recurrent connection per type, turn off recurrent training
        # of individual connections
        if train_recurrent:
            if train_recurrent_per_type:
                individual_training = False
                per_type_training = True
            else:
                individual_training = True
                per_type_training = False
        else:
            individual_training = False
            per_type_training = False

        # Scale the weights
        self.recurrent_weight_values = tf.Variable(
            weights * recurrent_weight_scale / lr_scale, 
            name="sparse_recurrent_weights",
            constraint=SignedConstraint(recurrent_weight_positive),
            trainable=individual_training,
            dtype=self.variable_dtype
        ) # shape = (n_synapses,)

        # prepare per_type variable, if required
        if per_type_training:
            self.per_type_training = True
            self.connection_type_ids = other_v1_utils.connection_type_ids(network)
            max_id = np.max(self.connection_type_ids) + 1
            # prepare a variable and gather with type ids.
            self.recurrent_per_type_weight_values = tf.Variable(
                tf.ones(max_id),
                name="recurrent_per_type_weights",
                constraint=ClipConstraint(0.2, 5.0),
                trainable=True,
                dtype=self.variable_dtype
            ) # shape = (n_connection_types (21 * 21))
            # multiply this to the weights (this needs to be done in the loop)
        else:
            self.per_type_training = False
            
        self.syn_ids = tf.constant(syn_ids, dtype=tf.int64)
        # self.recurrent_weights_factors = tf.gather(self.synaptic_basis_weights, self.syn_ids, axis=0) # TensorShape([23525415, 5])
        print(f"    > # Recurrent synapses: {len(indices)}")

        del indices, weights, dense_shape, delays, syn_ids

        ### LGN input connectivity ###
        self.input_dim = lgn_input["n_inputs"]
        self.lgn_input_dense_shape = (self._n_neurons, self.input_dim,)
        input_indices = np.array(lgn_input["indices"])
        input_weights = np.array(lgn_input["weights"])
        input_syn_ids = np.array(lgn_input["syn_ids"])
        input_delays = np.array(lgn_input["delays"])
        # Scale down the input weights
        input_weights = (input_weights/ voltage_scale[self._node_type_ids[input_indices[:, 0]]])
        input_delays = np.round(np.clip(input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the postsynaptic neuron indices
        # input_indices[:, 1] = input_indices[:, 1] + self._n_neurons * (input_delays - 1)
        self.input_indices = tf.Variable(input_indices, trainable=False, dtype=tf.int64)

        # Define the Tensorflow variables
        # input_weights = tf.expand_dims(input_weights, axis=1) # add dimension for the weights factors - TensorShape([23525415, 1])
        # input_weight_positive = tf.Variable(
        #     input_weights >= 0.0, name="input_weights_sign", trainable=False)
        input_weight_positive = tf.constant(input_weights >= 0, dtype=tf.int8)

        self.input_weight_values = tf.Variable(
            input_weights * input_weight_scale / lr_scale,
            name="sparse_input_weights",
            constraint=SignedConstraint(input_weight_positive),
            trainable=train_input,
            dtype=self.variable_dtype
        )
        self.input_syn_ids = tf.constant(input_syn_ids, dtype=tf.int64)
        if not self._current_input:
            self.pre_input_ind_table = make_pre_ind_table(input_indices, n_source_neurons=self.input_dim)

        print(f"    > # LGN input synapses {len(input_indices)}")
        del input_indices, input_weights, input_syn_ids, input_delays

        ### BKG input connectivity ###
        self.bkg_input_dense_shape = (self._n_neurons, bkg_input["n_inputs"],)
        bkg_input_indices = np.array(bkg_input['indices'])
        bkg_input_weights = np.array(bkg_input['weights'])
        bkg_input_syn_ids = np.array(bkg_input['syn_ids'])
        bkg_input_delays = np.array(bkg_input['delays'])

        bkg_input_weights = (bkg_input_weights/voltage_scale[self._node_type_ids[bkg_input_indices[:, 0]]])
        bkg_input_delays = np.round(np.clip(bkg_input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the postsynaptic neuron indices
        # bkg_input_indices[:, 1] = bkg_input_indices[:, 1] + self._n_neurons * (bkg_input_delays - 1)
        self.bkg_input_indices = tf.Variable(bkg_input_indices, trainable=False, dtype=tf.int64)
        self.pre_bkg_ind_table = make_pre_ind_table(bkg_input_indices, n_source_neurons=bkg_input["n_inputs"])

        # Define Tensorflow variables
        # bkg_input_weight_positive = tf.Variable(
        #     bkg_input_weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
        bkg_input_weight_positive = tf.constant(bkg_input_weights >= 0, dtype=tf.int8)
        self.bkg_input_weights = tf.Variable(
            bkg_input_weights * input_weight_scale / lr_scale, 
            name="rest_of_brain_weights", 
            constraint=SignedConstraint(bkg_input_weight_positive),
            trainable=train_noise,
            dtype=self.variable_dtype
        )

        self.bkg_input_syn_ids = tf.constant(bkg_input_syn_ids, dtype=tf.int64)
        # self.bkg_input_weights_factors = tf.gather(self.synaptic_basis_weights, bkg_input_syn_ids, axis=0)
        
        print(f"    > # BKG input synapses {len(bkg_input_indices)}")
        del bkg_input_indices, bkg_input_weights, bkg_input_syn_ids, bkg_input_delays

    def calculate_input_current_from_spikes(self, x_t):
        # x_t: Shape [batch_size, input_dim]
        # batch_size = tf.shape(x_t)[0]
        n_post_neurons = self.lgn_input_dense_shape[0]
        # Find the indices of non-zero inputs
        non_zero_indices = tf.where(x_t > 0)
        batch_indices = non_zero_indices[:, 0]
        pre_neuron_indices = non_zero_indices[:, 1]
        # Get the indices into self.recurrent_indices for each pre_neuron_index
        # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
        new_indices, new_weights, new_syn_ids, post_in_degree, _ = get_new_inds_table(self.input_indices, self.input_weight_values, self.input_syn_ids, pre_neuron_indices, self.pre_input_ind_table)
        # Expand batch_indices to match the length of inds_flat
        # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
        batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
        # Get post-synaptic neuron indices
        post_neuron_indices = new_indices[:, 0]        
        # Compute segment IDs
        segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
        num_segments = self.batch_size * n_post_neurons  
        # Get the weights for each active synapse
        new_weights = tf.expand_dims(new_weights, axis=1)
        new_weights = new_weights * tf.gather(self.synaptic_basis_weights, new_syn_ids, axis=0)
        # Calculate input currents
        i_in_flat = tf.math.unsorted_segment_sum(
            new_weights,
            segment_ids,
            num_segments=num_segments
        )
        # Cast i_rec to the compute dtype if necessary
        if i_in_flat.dtype != self.compute_dtype:
            i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

        # Add batch dimension
        # i_in_flat = tf.reshape(i_in_flat, [batch_size, -1])

        return i_in_flat
    
    def calculate_input_current_from_firing_probabilities(self, x_t):
        """
        Calculate the input current to the LGN neurons from the input layer.
        """
        # batch_size = tf.shape(x_t)[0]
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        i_in = tf.TensorArray(dtype=self.compute_dtype, size=self._n_syn_basis)
        for r_id in range(self._n_syn_basis):
            input_weights_factors = tf.gather(self.synaptic_basis_weights[:, r_id], self.input_syn_ids, axis=0)
            weights_syn_receptors = self.input_weight_values * input_weights_factors
            sparse_w_in = tf.sparse.SparseTensor(
                self.input_indices,
                weights_syn_receptors,
                self.lgn_input_dense_shape
            )
            i_receptor = tf.sparse.sparse_dense_matmul(
                                                sparse_w_in, 
                                                tf.cast(x_t, dtype=self.variable_dtype), 
                                                adjoint_b=True
                                                )
            # Optionally cast the output back to float16
            if i_receptor.dtype != self.compute_dtype:
                i_receptor = tf.cast(i_receptor, dtype=self.compute_dtype)
            # Append i_receptor to the TensorArray
            i_in = i_in.write(r_id, i_receptor)
        # Stack the TensorArray into a single tensor
        i_in = i_in.stack()
        # flat the output
        i_in = tf.transpose(i_in)
        i_in_flat = tf.reshape(i_in, [self.batch_size * self._n_neurons, self._n_syn_basis])

        return i_in_flat
    
    
    def calculate_noise_current(self, rest_of_brain):
        # x_t: Shape [batch_size, input_dim]
        batch_size = tf.shape(rest_of_brain)[0]
        n_post_neurons = self.bkg_input_dense_shape[0]
        # Find the indices of non-zero inputs
        non_zero_indices = tf.where(rest_of_brain > 0)
        batch_indices = non_zero_indices[:, 0]
        pre_neuron_indices = non_zero_indices[:, 1]
        # Get the indices into self.recurrent_indices for each pre_neuron_index
        # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
        new_indices, new_weights, new_syn_ids, post_in_degree, _ = get_new_inds_table(self.bkg_input_indices, self.bkg_input_weights, self.bkg_input_syn_ids, pre_neuron_indices, self.pre_bkg_ind_table)
        # Expand batch_indices to match the length of inds_flat
        # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
        batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
        # Get post-synaptic neuron indices
        post_neuron_indices = new_indices[:, 0]        
        # Compute segment IDs
        segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
        num_segments = batch_size * n_post_neurons  
        # Get the number of presynaptic spikes
        # n_pre_spikes = tf.cast(tf.gather(rest_of_brain[0, :], new_indices[:, 1]), dtype=self.variable_dtype)
        # Get the number of presynaptic spikes
        presynaptic_indices = tf.stack([batch_indices_per_connection, new_indices[:, 1]], axis=1)
        n_pre_spikes = tf.cast(tf.gather_nd(rest_of_brain, presynaptic_indices), dtype=self.variable_dtype)
        # Get the weights for each active synapse
        new_weights = tf.expand_dims(new_weights * n_pre_spikes, axis=1)
        new_weights = new_weights * tf.gather(self.synaptic_basis_weights, new_syn_ids, axis=0)
        # Calculate input currents
        i_in_flat = tf.math.unsorted_segment_sum(
            new_weights,
            segment_ids,
            num_segments=num_segments
        )
        # Cast i_rec to the compute dtype if necessary
        if i_in_flat.dtype != self.compute_dtype:
            i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

        return i_in_flat
    
    def calculate_i_rec_with_custom_grad(self, rec_z_buf):     
        # # Replace the original function call with:
        # i_rec_flat = calculate_synaptic_currents_cuda(
        #     tf.cast(rec_z_buf, dtype=self.variable_dtype), 
        #     self.recurrent_indices, 
        #     self.recurrent_weight_values, 
        #     self.recurrent_dense_shape, 
        #     self.synaptic_basis_weights, 
        #     tf.cast(self.syn_ids, dtype=tf.int32), 
        #     self.pre_ind_table
        # )     
        i_rec_flat = calculate_synaptic_currents(rec_z_buf, self.recurrent_indices, self.recurrent_weight_values, self.recurrent_dense_shape, 
                                                 self.synaptic_basis_weights, self.syn_ids, self.pre_ind_table)
        # # Cast i_rec to the compute dtype if necessary
        if i_rec_flat.dtype != self.compute_dtype:
            i_rec_flat = tf.cast(i_rec_flat, dtype=self.compute_dtype)

        return i_rec_flat
    
    def update_psc(self, psc, psc_rise, rec_inputs):
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        return new_psc, new_psc_rise

    @property
    def state_size(self):
        # Define the state size of the network
        state_size = (
            self._n_neurons * self.max_delay,  # z buffer
            self._n_neurons,  # v
            self._n_neurons,  # r
            self._n_neurons * 2,  # asc
            self._n_neurons * self._n_syn_basis,  # psc rise
            self._n_neurons * self._n_syn_basis,  # psc
        )
        return state_size
    
    def zero_state(self, batch_size, dtype=tf.float32):
        # The neurons membrane voltage start the simulation at their reset value
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * tf.cast(
            self.v_th * 0.0 + 1.0 * self.v_reset, dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc = tf.zeros((batch_size, self._n_neurons * 2), dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), dtype)

        return z0_buf, v0, r0, asc, psc_rise0, psc0

    # @tf.function # dont use it in here because it breaks the graph structure and the custom gradients
    def call(self, inputs, state, constants=None):

        # Get all the model inputs
        # external_current = inputs[:, :self._n_neurons*self._n_syn_basis] # external inputs shape (1, 399804)
        # bkg_noise = inputs[:, self._n_neurons*self._n_syn_basis:-self._n_neurons]
        lgn_input = inputs[:, :self.input_dim]
        # bkg_input = inputs[:, self.input_dim:-self._n_neurons]
        state_input = inputs[:, -self._n_neurons:] # dummy zeros
        # batch_size = tf.shape(lgn_input)[0]

        bkg_input = tf.random.poisson(shape=(self.batch_size, self.bkg_input_dense_shape[1]), 
                                    lam=self._bkg_firing_rate*.001, 
                                    dtype=self.variable_dtype) # this implementation is slower

        if self._spike_gradient:
            state_input = tf.zeros((1,), dtype=self.compute_dtype)
        else:
            state_input = tf.zeros((4,), dtype=self.compute_dtype)
                
        # Extract the network variables from the state
        z_buf, v, r, asc, psc_rise, psc = state
        # Get previous spikes
        prev_z = z_buf[:, :self._n_neurons]  # Shape: [batch_size, n_neurons]
        # Define the spikes buffer
        dampened_z_buf = z_buf * self._recurrent_dampening  # dampened version of z_buf 
        # Now we use tf.stop_gradient to prevent the term (z_buf - dampened_z_buf) to be trained
        rec_z_buf = (tf.stop_gradient(z_buf - dampened_z_buf) + dampened_z_buf)  
        # Calculate the recurrent postsynaptic currents
        i_rec = self.calculate_i_rec_with_custom_grad(rec_z_buf)
        # Calculate the postsynaptic current from the external input
        if self._current_input:
            external_current = self.calculate_input_current_from_firing_probabilities(lgn_input)
        else:
            external_current = self.calculate_input_current_from_spikes(lgn_input)

        i_noise = self.calculate_noise_current(bkg_input)
        # Add all the current sources
        rec_inputs = i_rec + external_current + i_noise
        # Reshape i_rec_flat back to [batch_size, num_neurons]
        rec_inputs = tf.reshape(rec_inputs, [self.batch_size, self._n_neurons * self._n_syn_basis])
        # Scale with the learning rate
        rec_inputs = rec_inputs * self._lr_scale
        if constants is not None and not self._spike_gradient:
            rec_inputs = rec_inputs + state_input * self._lr_scale

        # Calculate the new psc variables
        new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)        
        # Calculate the ASC
        asc = tf.reshape(asc, (self.batch_size, self._n_neurons, 2))
        new_asc = self.exp_dt_k * asc + tf.expand_dims(prev_z, axis=-1) * self.asc_amps
        new_asc = tf.reshape(new_asc, (self.batch_size, self._n_neurons * 2))
        # Calculate the postsynaptic current 
        input_current = tf.reshape(psc, (self.batch_size, self._n_neurons, self._n_syn_basis))
        input_current = tf.reduce_sum(input_current, -1) # sum over receptors
        if constants is not None and self._spike_gradient:
            input_current += state_input

        # Add all the postsynaptic current sources
        c1 = input_current + tf.reduce_sum(asc, axis=-1) + self.gathered_g

        # Calculate the new voltage values
        decayed_v = self.decay * v
        reset_current = prev_z * self.v_gap
        new_v = decayed_v + self.current_factor * c1 + reset_current

        # Update the voltage according to the LIF equation and the refractory period
        # New r is a variable that accounts for the refractory period in which a neuron cannot spike
        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)  # =max(r + prev_z * self.t_ref - self._dt, 0)
        if self._hard_reset:
            # Here we keep the voltage at the reset value during the refractory period
            new_v = tf.where(new_r > 0.0, self.v_reset, new_v)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period

        # Generate the network spikes
        v_sc = (new_v - self.v_th) / self.normalizer
        if self._pseudo_gauss:
            if self.compute_dtype == tf.bfloat16:
                new_z = spike_gauss_b16(v_sc, self._gauss_std, self._dampening_factor)
            elif self.compute_dtype == tf.float16:
                new_z = spike_gauss_16(v_sc, self._gauss_std, self._dampening_factor)
            else:
                new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)
        else:
            if self.compute_dtype == tf.bfloat16:
                new_z = spike_function_b16(v_sc, self._dampening_factor)
            elif self.compute_dtype == tf.float16:
                new_z = spike_function_16(v_sc, self._dampening_factor)
            else:
                new_z = spike_function(v_sc, self._dampening_factor)

        # Generate the new spikes if the refractory period is concluded
        new_z = tf.where(tf.greater(new_r, 0.0), tf.zeros_like(new_z), new_z)
        # Add current spikes to the buffer
        new_z_buf = tf.concat([new_z, z_buf[:, :-self._n_neurons]], axis=1)  # Shift buffer

        # Define the model outputs and the new state of the network
        outputs = (
            new_z,
            new_v,
            # new_v * self.voltage_scale + self.voltage_offset,
            # (input_current + tf.reduce_sum(asc, axis=-1)) * self.voltage_scale,
        )

        new_state = (
            new_z_buf,
            new_v,
            new_r,
            new_asc,
            new_psc_rise,
            new_psc,
        )

        return outputs, new_state


# @profile
def create_model(
    network,
    lgn_input,
    bkg_input,
    seq_len=100,
    n_input=10,
    n_output=2,
    dtype=tf.float32,
    input_weight_scale=1.0,
    gauss_std=0.5,
    dampening_factor=0.2,
    recurrent_dampening_factor=0.5,
    lr_scale=800.0,
    train_recurrent=True,
    train_recurrent_per_type=False,
    train_input=True,
    train_noise=True,
    neuron_output=False,
    use_state_input=False,
    return_state=False,
    return_sequences=False,
    down_sample=50,
    cue_duration=20,
    add_metric=True,
    max_delay=5,
    batch_size=None,
    pseudo_gauss=False,
    hard_reset=False,
    current_input=False
):

    # Create the input layer of the model
    x = tf.keras.layers.Input(shape=(None, n_input,))
    neurons = network["n_nodes"]

    # Create an input layer for the initial state of the RNN
    state_input_holder = tf.keras.layers.Input(shape=(None, neurons))
    state_input = tf.cast(tf.identity(state_input_holder), dtype)  

    # If batch_size is not provided as an argument, it is automatically inferred from the
    # first dimension of x using tf.shape().
    if batch_size is None:
        batch_size = tf.shape(x)[0]
    else:
        batch_size = batch_size

    # Create the V1Column cell
    print('Creating the V1 column...')
    # time0 = time()
    cell = V1Column(
        network,
        lgn_input,
        bkg_input,
        gauss_std=gauss_std,
        dampening_factor=dampening_factor,
        input_weight_scale=input_weight_scale,
        lr_scale=lr_scale,
        spike_gradient=True,
        recurrent_dampening_factor=recurrent_dampening_factor,
        max_delay=max_delay,
        pseudo_gauss=pseudo_gauss,
        batch_size=batch_size, 
        train_recurrent=train_recurrent,
        train_recurrent_per_type=train_recurrent_per_type,
        train_input=train_input,
        train_noise=train_noise,
        hard_reset=hard_reset,
        current_input=current_input
    )

    # initialize the RNN state to zero using the zero_state() method of the V1Column class.
    zero_state = cell.zero_state(batch_size, dtype)

    if use_state_input:
        # The shape of each input tensor matches the shape of the corresponding
        # tensor in the zero_state tuple, except for the batch dimension. The batch
        # dimension is left unspecified, allowing the tensor to be fed variable-sized
        # batches of data.
        initial_state_holder = tf.nest.map_structure(lambda _x: tf.keras.layers.Input(shape=_x.shape[1:], dtype=_x.dtype), zero_state)
        # The code then copies the input tensors into the rnn_initial_state variable
        # using tf.nest.map_structure(). This creates a nested structure of tensors with
        # the same shape as the original zero_state structure.
        rnn_initial_state = tf.nest.map_structure(tf.identity, initial_state_holder)
        # In both cases, the code creates a constants tensor using tf.zeros_like() or
        # tf.zeros(). This tensor is used to provide constant input to the RNN during
        # computation. The shape of the constants tensor matches the batch_size.
        constants = tf.zeros_like(rnn_initial_state[0][:, 0], dtype)
    else:
        rnn_initial_state = zero_state
        constants = tf.zeros((batch_size,))

    # Concatenate the input layer with the initial state of the RNN
    # full_inputs = tf.concat((tf.cast(x, dtype), bkg_inputs, state_input), -1) # (None, 600, 5*n_neurons+n_neurons)
    full_inputs = tf.concat((tf.cast(x, dtype), state_input), -1)
    
    # Create the RNN layer of the model using the V1Column cell
    # The RNN layer returns the output of the RNN layer and the final state of the RNN
    # layer. The output of the RNN layer is a tensor of shape (batch_size, seq_len,
    # neurons). The final state of the RNN layer is a tuple of tensors, each of shape
    # (batch_size, neurons).
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=return_state, name="rsnn")

    # Apply the rnn layer to the full_inputs tensor
    rsnn_out = rnn(full_inputs, initial_state=rnn_initial_state, constants=constants)

    # Check if the return_state argument is True or False and assign the output of the
    # RNN layer to the hidden variable accordingly.
    if return_state:
        hidden = rsnn_out[0]
        # new_state = out[1:]
    else:
        hidden = rsnn_out

    spikes = hidden[0]
    # voltage = hidden[1]

    # computes the mean of the spikes tensor along the second and third dimensions
    # (which represent time and neurons),
    rate = tf.reduce_mean(spikes)

    # The neuron output option selects only the output neurons from the spikes tensor
    if neuron_output:
        # The output_spikes tensor is computed by taking a linear combination
        # of the current spikes and the previous spikes, with the coefficients
        # determined by the dampening_factor. This serves to reduce the volatility
        # of the output spikes, making them more stable.
        output_spikes = 1 / dampening_factor * spikes + (1 - 1 / dampening_factor) * tf.stop_gradient(spikes)
        # The output tensor is then computed by selecting the spikes from the
        # output neurons and scaling them by a learned factor. The scale factor
        # is computed using a softplus activation function applied to the output
        # of a dense layer, and the threshold is computed by passing the output
        # spikes through another dense layer.
        output = tf.gather(output_spikes, network["readout_neuron_ids"], axis=2)
        output = tf.reduce_mean(output, -1)
        scale = 1 + tf.nn.softplus(tf.keras.layers.Dense(1)(tf.zeros_like(output[:1, :1])))
        thresh = tf.keras.layers.Dense(1)(tf.zeros_like(output))
        output = tf.stack([thresh[..., 0], output[..., -1]], -1) * scale
    # If neuron_output is False, then the output tensor is simply the result of
    # passing the spikes tensor through a dense layer with n_output units.
    else:
        output = tf.keras.layers.Dense(n_output, name="projection", trainable=False)(spikes)
        # output = tf.keras.layers.Dense(n_output, name="projection", trainable=True)(spikes)
        
    # Finally, a prediction layer is created
    # output = tf.keras.layers.Lambda(lambda _a: _a, name="prediction")(output)

    # If return_sequences is True, then the mean_output tensor is computed by
    # averaging over sequences of length down_sample in the output tensor.
    # Otherwise, mean_output is simply the mean of the last cue_duration time steps
    # of the output tensor.
    # if return_sequences:
    #     mean_output = tf.reshape(output, (-1, int(seq_len / down_sample), down_sample, n_output))
    #     mean_output = tf.reduce_mean(mean_output, 2)
    #     mean_output = tf.nn.softmax(mean_output, axis=-1)
    # else:
    #     mean_output = tf.reduce_mean(output[:, -cue_duration:], 1)
    #     mean_output = tf.nn.softmax(mean_output)

    if use_state_input:
        # many_input_model = tf.keras.Model(
        #     inputs=[x, state_input_holder, initial_state_holder], 
        #     outputs=mean_output
        # )
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder,initial_state_holder],
            outputs=[output])
    else:
        # many_input_model = tf.keras.Model(
        #     inputs=[x, state_input_holder], 
        #     outputs=mean_output
        # )
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder],
            outputs=[output])

    if add_metric:
        # add the firing rate of the neurons as a metric to the model
        many_input_model.add_metric(rate, name="rate")

    return many_input_model


# if name is main run the create model function
if __name__ == "__main__":
    # load the network
    import load_sparse

    n_input = 17400
    n_neurons = 1574

    network, lgn_input, bkg_input = load_sparse.cached_load_v1(
        n_input,
        n_neurons,
        True,
        "GLIF_network",
        seed=3000,
        connected_selection=False,
        n_output=2,
        neurons_per_output=16,
    )
    # create the model
    model = create_model(
        network,
        lgn_input,
        bkg_input,
        seq_len=100,
        n_input=n_input,
        n_output=2,
        cue_duration=20,
        dtype=tf.float32,
        input_weight_scale=1.0,
        gauss_std=0.5,
        dampening_factor=0.2,
        lr_scale=800.0,
        train_recurrent=True,
        train_recurrent_per_type=False,
        train_input=True,
        neuron_output=False,
        recurrent_dampening_factor=0.5,
        use_state_input=False,
        return_state=False,
        return_sequences=False,
        down_sample=50,
        add_metric=True,
        max_delay=5,
        batch_size=1,
        pseudo_gauss=False,
        hard_reset=True,
    )
