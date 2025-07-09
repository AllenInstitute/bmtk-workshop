# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:43:44 2022

@author: javig
"""


import pandas as pd
import os
import sys
import glob
import numpy as np
import tensorflow as tf
import h5py
import time
from scipy.ndimage import gaussian_filter1d
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management


def pop_name_to_cell_type(pop_name, ignore_l5e_subtypes=False):
    """convert pop_name in the old format to cell types.
    for example,
    'e4Rorb' -> 'L4 Exc'
    'i4Pvalb' -> 'L4 PV'
    'i23Sst' -> 'L2/3 SST'
    'e5ET' -> 'L5 ET'
    """
    shift = 0  # letter shift for L23
    layer = pop_name[1]
    if layer == "2":
        layer = "2/3"
        shift = 1
    elif layer == "1":
        return "L1 Htr3a"  # special case

    class_name = pop_name[2 + shift :]
    if class_name == "Pvalb":
        subclass = "PV"
    elif class_name == "Sst":
        subclass = "SST"
    elif (class_name == "Vip") or (class_name == "Htr3a"):
        subclass = "VIP"
    else:  # excitatory
        if layer == "5" and not ignore_l5e_subtypes:
            subclass = class_name
        else:
            subclass = "Exc"

    return f"L{layer} {subclass}"  

def get_layer_info(network):
    pop_name = pop_names(network)
    layer_query = ["e23", "e4", "e5", "e6"]
    layer_names = ["EXC_L23", "EXC_L4", "EXC_L5", "EXC_L6"]
    layer_info = {}
    for i in range(4):
        layer_info[layer_names[i]] = np.char.startswith(pop_name, layer_query[i])
    return layer_info  

def pop_names(network, core_radius = None, n_selected_neurons=None, data_dir='', return_node_type_ids=False):
    if data_dir != '':  # if changed from default, use as is.
        pass
    elif "data_dir" in network:  # if defined in the network, use it.
        data_dir = network["data_dir"]
    else:
        print("No data_dir defined in the network. Using the default one.")
        data_dir = 'GLIF_network'  # if none is the cae, use the default one
    path_to_csv = os.path.join(data_dir, 'network/v1_node_types.csv')
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')

    # Read data
    node_types = pd.read_csv(path_to_csv, sep=' ')
    with h5py.File(path_to_h5, mode='r') as node_h5:
        # Create mapping from node_type_id to pop_name
        node_types.set_index('node_type_id', inplace=True)
        node_type_id_to_pop_name = node_types['pop_name'].to_dict()

        # Map node_type_id to pop_name for all neurons and select population names of neurons in the present network 
        node_type_ids = np.array(node_h5['nodes']['v1']['node_type_id'][()])[network['tf_id_to_bmtk_id']]
        true_pop_names = np.array([node_type_id_to_pop_name[nid] for nid in node_type_ids])

    if core_radius is not None:
        selected_mask = isolate_core_neurons(network, radius=core_radius, data_dir=data_dir)
    elif n_selected_neurons is not None:
        selected_mask = isolate_core_neurons(network, n_selected_neurons=n_selected_neurons, data_dir=data_dir)
    else:
        selected_mask = np.full(len(true_pop_names), True)
        
    true_pop_names = true_pop_names[selected_mask]
    node_type_ids = node_type_ids[selected_mask]

    if return_node_type_ids:
        return true_pop_names, node_type_ids
    else:
        return true_pop_names



    
    

def connection_type_ids(network, core_radius=None, data_dir='', return_names=False):
    # first, get the pop_names
    pop_names_var = pop_names(network, core_radius=core_radius, data_dir=data_dir)
    cell_types = [pop_name_to_cell_type(pop_name) for pop_name in pop_names_var]
    
    # make an inverse index of the pop_names
    all_names, pop_ids_cells = np.unique(cell_types, return_inverse=True)
    
    # get the pre and post cells
    pre_cells = network["synapses"]["indices"][:, 0]
    post_cells = network["synapses"]["indices"][:, 1] % network["n_nodes"]
    
    # make a unique number for the connection from each type to another type
    pop_ids_synapses = pop_ids_cells[pre_cells] * 1000 + pop_ids_cells[post_cells]
    
    # make an inverse dictionary of the ids. This defines connection type_ids.
    all_pop_ids, connection_type_ids = np.unique(pop_ids_synapses, return_inverse=True)
    
    if return_names:
        # decode the names from the all_pop_ids
        names = {}
        names["pre_id"] = all_pop_ids//1000
        names["pre"] = all_names[names["pre_id"]]
        names["post_id"] = all_pop_ids%1000
        names["post"] = all_names[names["post_id"]]
        names["all_names"] = [f'{names["pre"][i]} -> {names["post"][i]}' for i in range(len(all_pop_ids))]
        return connection_type_ids, names
    
    return connection_type_ids

def angle_tunning(network, data_dir='GLIF_network'):
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    with h5py.File(path_to_h5, mode='r') as node_h5:
        angle_tunning = np.array(node_h5['nodes']['v1']['0']['tuning_angle'][:])[network['tf_id_to_bmtk_id']]
    
    return angle_tunning

def isolate_core_neurons(network, radius=None, n_selected_neurons=None, data_dir='GLIF_network'):
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    with h5py.File(path_to_h5, mode='r') as node_h5:
        x = np.array(node_h5['nodes']['v1']['0']['x'][()])[network['tf_id_to_bmtk_id']]
        z = np.array(node_h5['nodes']['v1']['0']['z'][()])[network['tf_id_to_bmtk_id']]
        
    r = np.sqrt(x ** 2 + z ** 2)
    if radius is not None:
        selected_mask = r < radius
    elif n_selected_neurons is not None:
        selected_mask = np.argsort(r)[:n_selected_neurons]
        selected_mask = np.isin(np.arange(len(r)), selected_mask)
    
    return selected_mask
    
def isolate_neurons(network, neuron_population='e23', data_dir='GLIF_network'):
    n_neurons = network['n_nodes']
    node_types_path = os.path.join(data_dir, 'network/v1_node_types.csv')
    node_types = pd.read_csv(node_types_path, sep=' ')

    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')

    with h5py.File(path_to_h5, mode='r') as node_h5:
        # Create mapping from node_type_id to pop_name
        node_types.set_index('node_type_id', inplace=True)
        node_type_id_to_pop_name = node_types['pop_name'].to_dict()

        # Get node_type_ids for the current network
        node_type_ids = np.array(node_h5['nodes']['v1']['node_type_id'][()])
        true_node_type_ids = node_type_ids[network['tf_id_to_bmtk_id']]
        selected_mask = np.zeros(n_neurons, bool)

        # Vectorize the selection of neurons based on population
        for pop_id, pop_name in node_type_id_to_pop_name.items():
            # if pop_name[0] == neuron_population[0] and pop_name[1] == neuron_population[1]:
            if neuron_population in pop_name:
                # choose all the neurons of the given pop_id
                sel = true_node_type_ids == pop_id
                selected_mask = np.logical_or(selected_mask, sel)

    return selected_mask


    
def firing_rates_smoothing(z, sampling_rate=60, window_size=100): #window_size=300
    n_simulations, simulation_length, n_neurons = z.shape
    sampling_interval = int(1000/sampling_rate) #ms
    window_size = int(np.round(window_size/sampling_interval))
    #z = z.reshape(n_simulations, simulation_length, z.shape[1])
    z_chunks = [z[:, x:x+sampling_interval, :] for x in range(0, simulation_length, sampling_interval)]
    sampled_firing_rates = np.array([np.sum(group, axis = 1) * sampling_rate for group in z_chunks])  # (simulation_length, n_simulations, n_neurons)
    smoothed_fr = gaussian_filter1d(sampled_firing_rates, window_size, axis=0)
    smoothed_fr = np.swapaxes(smoothed_fr, 0, 1)
    return smoothed_fr, sampling_interval

def voltage_spike_effect_correction(v, z, pre_spike_gap=2, post_spike_gap=3):
    n_simulations, simulation_length, n_neurons = v.shape
    v = v.reshape((n_simulations*simulation_length, n_neurons))
    z = z.reshape((n_simulations*simulation_length, n_neurons))
    # Find the spike times excluding the last miliseconds
    spikes_idx, neurons_idx = np.where(z[:-post_spike_gap,:]==1)
    # Filter early spikes
    mask = spikes_idx>= pre_spike_gap
    spikes_idx = spikes_idx[mask]
    neurons_idx = neurons_idx[mask]
    for t_idx, n_idx in zip(spikes_idx, neurons_idx):
        pre_t_idx = t_idx-pre_spike_gap
        post_t_idx = t_idx+post_spike_gap
        prev_value = v[pre_t_idx, n_idx]
        post_value = v[post_t_idx, n_idx]
        # Make a linear interpolation of the voltage in the surroundings of the spike
        step = (post_value-prev_value)/(pre_spike_gap+post_spike_gap+1)
        if step==0:
            new_values = np.ones(pre_spike_gap+post_spike_gap+1)*post_value
        else:
            new_values = np.arange(prev_value, post_value, step)
        v[pre_t_idx:post_t_idx+1, n_idx] = new_values
    v = v.reshape((n_simulations, simulation_length, n_neurons))
    return v


def optimizers_match(current_optimizer, checkpoint_directory):
    current_optimizer_vars = {v.name: v.shape.as_list() for v in current_optimizer.variables()}
    checkpoint_vars = tf.train.list_variables(checkpoint_directory)
    checkpoint_optimizer_vars = {name.split('/.ATTRIBUTES')[0]: value for name, value in checkpoint_vars if 'optimizer' in name}
    if 'optimizer/loss_scale/current_loss_scale' in checkpoint_optimizer_vars or 'optimizer/loss_scale/good_steps' in checkpoint_optimizer_vars:
        if len(current_optimizer_vars) != len(checkpoint_optimizer_vars)-3:
            print('Checkpoint optimizer variables do not match the current optimizer variables.. Renewing optimizer...')
            return False
        else:
            for name, desired_shape in current_optimizer_vars.items():
                var_not_matched = True
                for opt_var, opt_var_shape in checkpoint_optimizer_vars.items():
                    if opt_var_shape == desired_shape: 
                        var_not_matched = False
                        del checkpoint_optimizer_vars[opt_var]
                        break
                if var_not_matched:
                    print(f'{name} does not have a match')
                    return False
            return True
    else:
        if len(current_optimizer_vars) != len(checkpoint_optimizer_vars)-1:
            print('Checkpoint optimizer variables do not match the current optimizer variables.. Renewing optimizer...')
            return False
        else:
            for name, desired_shape in current_optimizer_vars.items():
                var_not_matched = True
                for opt_var, opt_var_shape in checkpoint_optimizer_vars.items():
                    if opt_var_shape == desired_shape: 
                        var_not_matched = False
                        del checkpoint_optimizer_vars[opt_var]
                        break
                if var_not_matched:
                    print(f'{name} does not have a match')
                    return False
            return True

############################ DATA SAVING AND LOADING METHODS #########################
class SaveSimDataHDF5:
    def __init__(self, flags, keys, data_path, network, save_core_only=True, dtype=np.float16):
        self.keys = keys
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        self.dtype = dtype
        if save_core_only:
            self.core_mask = isolate_core_neurons(network, data_dir=flags.data_dir)
        else:
            self.core_mask = np.full(flags.neurons, True)
        self.V1_data_shape = (flags.n_simulations, flags.seq_len, flags.neurons)
        self.V1_core_data_shape = (flags.n_simulations, flags.seq_len, self.core_mask.sum())
        self.LGN_data_shape = (flags.n_simulations, flags.seq_len, flags.n_input)
        with h5py.File(os.path.join(self.data_path, 'simulation_data.hdf5'), 'w') as f:
            g = f.create_group('Data')
            for key in self.keys:
                if key=='z':
                    g.create_dataset(key, self.V1_data_shape, dtype=np.uint8, 
                                     chunks=True, compression='gzip', shuffle=True)
                elif key=='z_lgn':
                    g.create_dataset(key, self.LGN_data_shape, dtype=np.uint8, 
                                     chunks=True, compression='gzip', shuffle=True)
                else:
                    g.create_dataset(key, self.V1_core_data_shape, dtype=self.dtype, 
                                     chunks=True, compression='gzip', shuffle=True)
            for flag, val in flags.flag_values_dict().items():
                if isinstance(val, (float, int, str, bool)):
                    g.attrs[flag] = val
            g.attrs['Date'] = time.time()
                
    def __call__(self, simulation_data, trial):
        with h5py.File(os.path.join(self.data_path, 'simulation_data.hdf5'), 'a') as f:
            for key, val in simulation_data.items():
                if key in ['z', 'z_lgn']:
                    val = np.array(val).astype(np.uint8)
                    # val = np.packbits(val)
                else:
                    val = np.array(val)[:, :, self.core_mask].astype(self.dtype)
                f['Data'][key][trial, :, :] = val
    
    
# def save_simulation_results_h5df(flags, simulation_data, network, data_path, trial, save_core_only=True,
#                             dtype=np.float16):
#     if save_core_only:
#         core_mask = isolate_core_neurons(network, data_dir=flags.data_dir)
#     else:
#         core_mask = np.full(flags.neurons, True)
        
#     with h5py.File(os.path.join(data_path, 'Simulation_data.h5'), "wb") as f:
#         for key, val in simulation_data.items():
#             if key in ['z', 'z_lgn']:
#                 val = np.array(val).astype(np.uint8)
#                 val = np.packbits(val)
#             else:
#                 val = np.array(val)[:, :, core_mask].astype(dtype)
#             key_grp = f.create_group(key)
#             key_grp.create_dataset(trial, data=val, compression='gzip')
        
    # for key, val in simulation_data.items():
    #     if key in ['z', 'z_lgn']:
    #         val = np.array(val).astype(np.uint8)
    #         val = np.packbits(val)
    #         file_management.save_lzma(val, f'{key}_{trial}.lzma', data_path)
    #     else:
    #         val = np.array(val)[:, :, core_mask].astype(dtype)
    #         file_management.save_lzma(val, f'{key}_{trial}.lzma', data_path)
            

class SaveSimData:
    def __init__(self, flags, keys, data_path, network, save_core_only=True, 
                 compress_data=True, dtype=np.float16):
        self.keys = keys
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        self.dtype = dtype
        if save_core_only:
            self.core_mask = isolate_core_neurons(network, data_dir=flags.data_dir)
        else:
            self.core_mask = np.full(flags.neurons, True)
        self.V1_data_shape = (flags.n_simulations, flags.seq_len, flags.neurons)
        self.V1_core_data_shape = (flags.n_simulations, flags.seq_len, self.core_mask.sum())
        self.LGN_data_shape = (flags.n_simulations, flags.seq_len, flags.n_input)
        if compress_data:
            self.save_method = file_management.save_lzma
        else:
            self.save_method = file_management.save_pickle
                
    def __call__(self, simulation_data, trial):
        for key, val in simulation_data.items():
            if key in ['z', 'z_lgn']:
                val = np.array(val).astype(np.uint8)
                # val = np.packbits(val)
            else:
                val = np.array(val)[:, :, self.core_mask].astype(self.dtype)
            self.save_method(val, f'{key}_{trial}', self.data_path)
            

# def save_simulation_results(flags, simulation_data, network, data_path, trial, save_core_only=True,
#                             compress_data=True, dtype=np.float16):
#     if save_core_only:
#         core_mask = isolate_core_neurons(network, data_dir=flags.data_dir)
#     else:
#         core_mask = np.full(flags.neurons, True)

#     if compress_data:
#         save_method = file_management.save_lzma
#     else:
#         save_method = file_management.save_pickle
        
#     for key, val in simulation_data.items():
#         if key in ['z', 'z_lgn']:
#             val = np.array(val).astype(np.uint8)
#             val = np.packbits(val)
#         else:
#             val = np.array(val)[:, :, core_mask].astype(dtype)
#         save_method(val, f'{key}_{trial}', data_path)


def load_simulation_results(full_data_path, n_simulations=None, skip_first_simulation=False, 
                            variables=None, simulation_length=2500, n_neurons=230924, 
                            n_core_neurons=51978, n_input=17400,
                            compress_data=True, dtype=np.float16):
    if compress_data:
        load_method = file_management.load_lzma
    else:
        load_method = file_management.load_pickle
    
    if n_simulations is None:
        n_simulations = len(glob.glob(os.path.join(full_data_path, 'v*')))
    first_simulation = 0
    last_simulation = n_simulations
    if skip_first_simulation:
        n_simulations -= 1
        first_simulation += 1
    if variables == None:
        variables = ['v', 'z', 'input_current', 'recurrent_current', 'bottom_up_current', 'z_lgn']
    if type(variables) == str:
        variables = [variables]
    data = {key: (np.empty((n_simulations, simulation_length, n_input), np.uint8) if key=='z_lgn' 
                  else np.empty((n_simulations, simulation_length, n_neurons), np.uint8) if key=='z' 
                  else np.empty((n_simulations, simulation_length, n_core_neurons), dtype))
            for key in variables}

    for i in range(first_simulation, last_simulation):
        for key, value in data.items():
            key_trial_file = glob.glob(os.path.join(full_data_path, f'{key}_{i}.*'))[0]
            data_array = load_method(key_trial_file)
            # if key == 'z':
                # unpacked_array = np.unpackbits(data_array)
                # data_array = unpacked_array.reshape((1,simulation_length,n_neurons))
            # elif key == 'z_lgn':
                # unpacked_array = np.unpackbits(data_array)
                # data_array = unpacked_array.reshape((1,simulation_length,n_input))
            if key in ['z', 'z_lgn']:
                data[key][(i-first_simulation):(i+1-first_simulation), :,:] = data_array.astype(np.uint8)
            else:
                data[key][(i-first_simulation):(i+1-first_simulation), :,:] = data_array.astype(np.float32)
            
    # if len(variables) == 1:
    #     data = data[key]
        
    return data, n_simulations


def load_simulation_results_hdf5(full_data_path, n_simulations=None, skip_first_simulation=False, 
                                variables=None):
    # Prepare dictionary to store the simulation metadata
    flags_dict = {}
    with h5py.File(full_data_path, 'r') as f:
        dataset = f['Data']
        flags_dict.update(dataset.attrs)
        # Get the simulation features
        if n_simulations is None:
            n_simulations = dataset['z'].shape[0]
        first_simulation = 0
        last_simulation = n_simulations
        if skip_first_simulation:
            n_simulations -= 1
            first_simulation += 1
        # Select the variables for the extraction
        if variables == None:
            variables = ['v', 'z', 'input_current', 'recurrent_current', 'bottom_up_current', 'z_lgn']
        if type(variables) == str:
            variables = [variables]
        # Extract the simulation data
        data = {}
        for key in variables:
            if key in ['z', 'z_lgn']:
               data[key] = np.array(dataset[key][first_simulation:last_simulation, :,:]).astype(np.uint8) 
            else:
                data[key] = np.array(dataset[key][first_simulation:last_simulation, :,:]).astype(np.float32)
            
    # if len(variables) == 1:
    #     data = data[key]
        
    return data, flags_dict, n_simulations