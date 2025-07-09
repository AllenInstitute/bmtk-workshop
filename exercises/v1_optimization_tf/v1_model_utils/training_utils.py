"""
Utility functions for V1 model training, refactored from multi_training_single_gpu_split.py.
"""
import os
import random
import numpy as np
import tensorflow as tf
from packaging import version
import absl # For flags, assuming it's used for config

from v1_model_utils import toolkit, other_v1_utils, load_sparse, models, loss_functions as losses # Assuming these are needed
from optimizers import ExponentiatedAdam
import stim_dataset # For dataset generation
from time import time
import copy # For deepcopying flags
import pickle as pkl # For loading EMA from checkpoint

class OptionFlags:
    def __init__(self):
        self.seed = 3000
        self.dtype = 'float16' # 'float16', 'bfloat16'
        self.ckpt_dir = '' # '/path/to/specific/checkpoint_run'
        self.results_dir = './results' # Base directory for results
        self.task_name = 'drifting_gratings_firing_rates_distr'
        self.neurons = 0
        self.n_input = 696 # Example, adjust as per actual flags
        self.core_only = False
        self.connected_selection = True
        self.random_weights = False
        self.uniform_weights = False
        self.optimizer = 'exp_adam'
        self.learning_rate = 0.005
        self.restore_from = '' # '/path/to/previous/run/Intermediate_checkpoints'
        self.batch_size = 1
        self.seq_len = 500
        self.n_output = 2
        self.cue_duration = 40
        self.input_weight_scale = 1.0
        self.dampening_factor = 0.5
        self.recurrent_dampening_factor = 0.5
        self.gauss_std = 0.3
        self.lr_scale = 1.
        self.train_input = False
        self.train_noise = True
        self.train_recurrent = True
        self.train_recurrent_per_type = False
        self.neuron_output = False
        self.pseudo_gauss = False
        self.hard_reset = False
        self.current_input = False
        self.max_delay = 5
        self.data_dir = 'GLIF_network'  # Updated data_dir
        self.bmtk_compat_lgn = False
        self.rotation = "ccw"
        self.caching = True # For load_sparse
        # Flags for losses and regularizers
        self.loss_core_radius = 100.0
        self.recurrent_weight_regularization = 1.
        self.uniform_weights = False # Affects rec_weight_regularizer
        self.recurrent_weight_regularizer_type = 'emd' # 'mean' or 'emd'
        self.all_neuron_rate_loss = False
        self.rate_cost = 10000.
        self.neuropixels_df = "Neuropixels_data/OSI_DSI_neuropixels_v4.csv" # Path to a dataframe or None
        self.voltage_cost = 1.0
        self.osi_cost = 20.
        self.osi_loss_method = 'crowd_osi' # 'gds', 'neuropixels_fr', etc.
        self.osi_loss_subtraction_ratio = 1.
        self.neurons_per_output = 16 # Added based on load_sparse.py requirement


        # To access default values like in absl.flags
        self._defaults = {
            'n_input': 0, 'core_only': False, 'connected_selection': False, 
            'random_weights': False, 'uniform_weights': False,
            'results_dir': '', 'task_name': 'default_task', 'ckpt_dir': '',
            'restore_from': '', 'batch_size': 1, 'seq_len': 500, 'n_output':2,
            'cue_duration':40, 'input_weight_scale':1.0, 'dampening_factor':0.5,
            'recurrent_dampening_factor':0.5, 'gauss_std':0.3, 'lr_scale':1.,
            'train_input':True, 'train_noise':True, 'train_recurrent':True,
            'train_recurrent_per_type':False, 'neuron_output':False, 'pseudo_gauss':False,
            'hard_reset':False, 'current_input':False, 'max_delay':5, 'data_dir':'GLIF_network',
            'bmtk_compat_lgn':False, 'rotation':0.0, 'caching':True,
            'loss_core_radius': 100.0, 'recurrent_weight_regularization': 1.0, 
            'recurrent_weight_regularizer_type': 'emd', 'all_neuron_rate_loss': True,
            'rate_cost': 10000.0, 'neuropixels_df': "Neuropixels_data/OSI_DSI_neuropixels_v4.csv", 'voltage_cost': 1.0, 'osi_cost': 20.0,
            'osi_loss_method': 'crowd_osi', 'osi_loss_subtraction_ratio': 1.0,
            'neurons_per_output': 16
        }
        # Ensure all attributes are in _defaults for __getitem__
        for key in self.__dict__.keys():
            if key not in self._defaults and key != '_defaults':
                self._defaults[key] = getattr(self,key) # Add with its own value as default

    def __getitem__(self, name):
        # Mocking the dictionary-like access for flags[name].default and flags[name].value
        class FlagValue:
            def __init__(self, value, default):
                self.value = value
                self.default = default
        return FlagValue(getattr(self, name), self._defaults.get(name))


def setup_environment(seed, tf_gpu_thread_mode='global', tf_cpp_min_log_level='0', tf_gpu_allocator='cuda_malloc_async'):
    """Sets up basic environment variables and seeds for reproducibility."""
    os.environ['TF_GPU_THREAD_MODE'] = tf_gpu_thread_mode
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_cpp_min_log_level
    os.environ['TF_GPU_ALLOCATOR'] = tf_gpu_allocator

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    physical_devices = tf.config.list_physical_devices("GPU")
    for dev in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(dev, True)
            print(f"Memory growth enabled for device {dev}")
        except RuntimeError:
            print(f"Memory growth already set or invalid device {dev}.")
    print(f"- Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n")

def configure_mixed_precision(dtype_str):
    """Configures mixed precision policy based on the dtype string."""
    dtype = tf.float32
    if dtype_str == 'float16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
            tf.keras.mixed_precision.experimental.set_policy(policy)
        else:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        dtype = tf.float16
        print('Mixed precision (float16) enabled!')
    elif dtype_str == 'bfloat16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_bfloat16")
            tf.keras.mixed_precision.experimental.set_policy(policy)
        else:
            tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        dtype = tf.bfloat16
        print('Mixed precision (bfloat16) enabled!')
    return dtype

def create_distribution_strategy(strategy_name="mirrored"):
    """Creates a TensorFlow distribution strategy."""
    if strategy_name == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
    elif strategy_name == "onedevice":
        device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
        strategy = tf.distribute.OneDeviceStrategy(device=device)
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")
    print(f"Using {strategy_name} strategy with {strategy.num_replicas_in_sync} replicas.")
    return strategy

def prepare_log_directory(flags, default_logdir_name='v1_model_training_logs'):
    """Prepares log directory for training. Uses absl.flags directly."""
    logdir = flags.ckpt_dir
    if not logdir or logdir == '':
        flag_components = [f'v1_{flags.neurons}']
        # Add other relevant flags to the name
        # This needs to be carefully curated based on common important flags
        for name in ['n_input', 'core_only', 'connected_selection', 'random_weights', 'uniform_weights']:
            if hasattr(flags, name) and flags[name].value != flags[name].default:
                flag_components.append(f'{name}_{flags[name].value}')
        
        flag_str = '_'.join(flag_components)
        results_dir = os.path.join(flags.results_dir if flags.results_dir else default_logdir_name, flag_str)
        os.makedirs(results_dir, exist_ok=True)
        print('Simulation results path: ', results_dir)
        sim_name = toolkit.get_random_identifier('b_') # Assuming toolkit is available
        logdir = os.path.join(results_dir, sim_name)
        print(f'> Results for {flags.task_name if hasattr(flags, "task_name") else "training"} will be stored in:\n {logdir} \n')
    else:
        print(f'> Using provided log directory: {logdir} \n')
        flag_str = logdir.split(os.path.sep)[-2] # To be consistent with original script

    os.makedirs(logdir, exist_ok=True)
    return logdir, flag_str

def create_optimizer(optimizer_name, learning_rate, dtype_str, epsilon=1e-11):
    """Creates an optimizer and wraps with LossScaleOptimizer if dtype is float16."""
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=epsilon)
    elif optimizer_name == 'exp_adam':
        optimizer = ExponentiatedAdam(learning_rate, epsilon=epsilon) # Make sure ExponentiatedAdam is correctly imported/defined
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.0, nesterov=False)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
    
    if dtype_str == 'float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        print("LossScaleOptimizer wrapped for float16 training.")
    return optimizer

def restore_checkpoint(model, optimizer, flags):
    """Restores model and optimizer from a checkpoint if specified in flags."""
    checkpoint_restored = False
    # Priority to ckpt_dir if it contains checkpoints
    potential_ckpt_path = os.path.join(flags.ckpt_dir, "Intermediate_checkpoints") if flags.ckpt_dir else ""
    if flags.ckpt_dir and os.path.exists(potential_ckpt_path) and tf.train.latest_checkpoint(potential_ckpt_path):
        checkpoint_directory = tf.train.latest_checkpoint(potential_ckpt_path)
        print(f'Attempting to restore checkpoint from ckpt_dir: {checkpoint_directory}...')
    # Fallback to restore_from if ckpt_dir is not valid or empty
    elif flags.restore_from and os.path.exists(flags.restore_from) and tf.train.latest_checkpoint(flags.restore_from):
        checkpoint_directory = tf.train.latest_checkpoint(flags.restore_from)
        print(f'Attempting to restore checkpoint from restore_from: {checkpoint_directory}...')
    else:
        print(f"No valid checkpoint found in {flags.ckpt_dir} or {flags.restore_from}. Starting from scratch...\n")
        return checkpoint_restored, optimizer # Return original optimizer

    # Create a new optimizer instance for comparison/restoration to avoid modifying the original one pre-emptively
    if flags.optimizer == 'adam':
        fresh_optimizer_for_restore = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)
    elif flags.optimizer == 'exp_adam':
        fresh_optimizer_for_restore = ExponentiatedAdam(flags.learning_rate, epsilon=1e-11)
    elif flags.optimizer == 'sgd':
        fresh_optimizer_for_restore = tf.keras.optimizers.SGD(flags.learning_rate, momentum=0.0, nesterov=False)
    else: # Should not happen if create_optimizer was called first
        raise ValueError(f"Invalid optimizer: {flags.optimizer}")

    if flags.dtype == 'float16':
        fresh_optimizer_for_restore = tf.keras.mixed_precision.LossScaleOptimizer(fresh_optimizer_for_restore)

    optimizer_matches = other_v1_utils.optimizers_match(fresh_optimizer_for_restore, checkpoint_directory)
    
    current_optimizer_to_use = optimizer # Default to the one passed in

    if not optimizer_matches:
        print(f"Optimizer in checkpoint does not match current configuration or is corrupted. Restoring model weights with a new optimizer.")
        # Use the fresh optimizer created for this scope
        if flags.optimizer == 'adam':
            current_optimizer_to_use = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)
        elif flags.optimizer == 'exp_adam':
            current_optimizer_to_use = ExponentiatedAdam(flags.learning_rate, epsilon=1e-11)
        elif flags.optimizer == 'sgd':
            current_optimizer_to_use = tf.keras.optimizers.SGD(flags.learning_rate, momentum=0.0, nesterov=False)
        
        if flags.dtype == 'float16':
             current_optimizer_to_use = tf.keras.mixed_precision.LossScaleOptimizer(current_optimizer_to_use)
        
        # Build optimizer before restoring if it's new
        if hasattr(model, 'trainable_variables') and model.trainable_variables:
            current_optimizer_to_use.build(model.trainable_variables)
        else: 
            print("Warning: Optimizer not built before checkpoint restoration as model has no trainable variables yet.")

        checkpoint = tf.train.Checkpoint(model=model, optimizer=current_optimizer_to_use) # Use the new optimizer
        try:
            # Use expect_partial for model weights only if optimizer is different or not being restored.
            status = checkpoint.restore(checkpoint_directory).expect_partial()
            # status.assert_existing_objects_matched() # Check if model weights were actually restored
            print('Checkpoint restored with a new optimizer (model weights only).')
            checkpoint_restored = True
        except Exception as e:
            print(f"Error during partial restoration with new optimizer: {e}")
    else:
        print(f"Optimizer in checkpoint matches. Restoring model and optimizer state.")
        # Build optimizer before restoring if it's the matched one
        if hasattr(model, 'trainable_variables') and model.trainable_variables:
            optimizer.build(model.trainable_variables) # Build the original optimizer passed in
        else: 
            print("Warning: Optimizer not built before checkpoint restoration as model has no trainable variables yet.")

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        try:
            checkpoint.restore(checkpoint_directory).assert_consumed()
            print('Checkpoint fully restored (model and optimizer)!')
            checkpoint_restored = True
            current_optimizer_to_use = optimizer # Confirm using the original, now restored, optimizer
        except Exception as e:
            print(f"Error during full restoration: {e}. Try expect_partial().")
            try:
                checkpoint.restore(checkpoint_directory).expect_partial()
                print('Checkpoint partially restored (model and optimizer).')
                checkpoint_restored = True
                current_optimizer_to_use = optimizer
            except Exception as e_partial:
                print(f"Error during partial restoration as fallback: {e_partial}")

    return checkpoint_restored, current_optimizer_to_use

def load_network_data(flags, flag_str):
    """Loads network, LGN input, and background input data."""
    t0 = time()
    if hasattr(flags, 'caching') and flags.caching:
        load_fn = load_sparse.cached_load_v1
    else:
        load_fn = load_sparse.load_v1
    network, lgn_input, bkg_input = load_fn(flags, flags.neurons, flag_str=flag_str)
    print(f"Model files loading: {time()-t0:.2f} seconds\n")
    return network, lgn_input, bkg_input

def create_v1_model(network, lgn_input, bkg_input, flags, dtype, per_replica_batch_size):
    """Creates the V1 model using specified configurations."""
    t0 = time()
    model = models.create_model(
        network,
        lgn_input,
        bkg_input,
        seq_len=flags.seq_len,
        n_input=flags.n_input,
        n_output=flags.n_output,
        cue_duration=flags.cue_duration,
        dtype=dtype,
        batch_size=per_replica_batch_size, # Use per_replica_batch_size for model definition
        input_weight_scale=flags.input_weight_scale,
        dampening_factor=flags.dampening_factor,
        recurrent_dampening_factor=flags.recurrent_dampening_factor,
        gauss_std=flags.gauss_std,
        lr_scale=flags.lr_scale,
        train_input=flags.train_input,
        train_noise=flags.train_noise,
        train_recurrent=flags.train_recurrent,
        train_recurrent_per_type=flags.train_recurrent_per_type,
        neuron_output=flags.neuron_output,
        pseudo_gauss=flags.pseudo_gauss,
        use_state_input=True,
        return_state=True,
        hard_reset=flags.hard_reset,
        add_metric=False, # Typically False for raw model, metrics handled in training loop
        max_delay=flags.max_delay if hasattr(flags, 'max_delay') else 5, # default from original
        current_input=flags.current_input
    )
    # Model build should happen within strategy.scope() after creation if not already built
    # For now, we assume it's built before use or the caller handles it.
    # model.build((per_replica_batch_size, flags.seq_len, flags.n_input))
    print(f"Model created in {time()-t0:.2f} s\n")
    return model

def get_gratings_dataset_fn(flags, global_batch_size, dtype, delays):
    """Returns a function that creates the drifting gratings dataset."""
    def _f(input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        _data_set = (stim_dataset.generate_drifting_grating_tuning(
            seq_len=flags.seq_len,
            pre_delay=delays[0],
            post_delay=delays[1],
            n_input=flags.n_input,
            data_dir=flags.data_dir,
            regular=False, # For training, typically not regular
            bmtk_compat=flags.bmtk_compat_lgn if hasattr(flags, 'bmtk_compat_lgn') else False,
            rotation=flags.rotation if hasattr(flags, 'rotation') else 0.0,
            dtype=dtype
        )
        .shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        )
        return _data_set
    return _f

def get_gray_dataset_fn(flags, global_batch_size, dtype):
    """Returns a function that creates the gray screen dataset (for spontaneous activity)."""
    def _f(input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        _gray_data_set = (stim_dataset.generate_drifting_grating_tuning(
            seq_len=flags.seq_len,
            pre_delay=flags.seq_len, # Full sequence is pre_delay for gray screen
            post_delay=0,
            n_input=flags.n_input,
            data_dir=flags.data_dir,
            rotation=flags.rotation if hasattr(flags, 'rotation') else 0.0,
            return_firing_rates=True, # Often used for spontaneous rate calculation
            dtype=dtype
        )
        .shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        )
        return _gray_data_set
    return _f

def setup_losses_and_regularizers(flags, model, network, load_fn, dtype, per_replica_batch_size, logdir, delays, flag_str=''):
    """Sets up various loss functions and regularizers for V1 model training."""
    print("Setting up losses and regularizers...")
    loss_components = {}

    # Create core and annulus masks
    if flags.loss_core_radius > 0:
        core_mask_np = other_v1_utils.isolate_core_neurons(network, radius=flags.loss_core_radius, data_dir=flags.data_dir)
        if core_mask_np.all():
            core_mask = None
            annulus_mask = None
            print("All neurons are in the core region. Core mask is set to None.")
        else:
            print(f"Core mask selects {core_mask_np.sum()} neurons.")
            core_mask = tf.constant(core_mask_np, dtype=tf.bool)
            annulus_mask = tf.constant(~core_mask_np, dtype=tf.bool)
    else:
        core_mask = None
        annulus_mask = None
    loss_components['core_mask'] = core_mask
    loss_components['annulus_mask'] = annulus_mask

    rsnn_layer = model.get_layer("rsnn")

    # Recurrent Weight Regularizer
    if flags.recurrent_weight_regularization > 0:
        rec_reg_network_data = network
        if flags.uniform_weights:
            print("Uniform weights are set. Loading network with original weights for recurrent regularizer.")
            dummy_flags = copy.deepcopy(flags)
            dummy_flags.uniform_weights = False 
            # Assuming load_fn can handle flags object directly for neuron count and other params
            rec_reg_network_data, _, _ = load_fn(dummy_flags, dummy_flags.neurons, flag_str='') # Use empty flag_str for generic load
        
        if flags.recurrent_weight_regularizer_type == 'mean':
            print("Using mean stiff recurrent weight regularizer")
            rec_weight_regularizer = losses.MeanStiffRegularizer(flags.recurrent_weight_regularization, rec_reg_network_data, penalize_relative_change=True, dtype=tf.float32)
        elif flags.recurrent_weight_regularizer_type == 'emd':
            print("Using EMD recurrent weight regularizer")
            rec_weight_regularizer = losses.EarthMoversDistanceRegularizer(flags.recurrent_weight_regularization, rec_reg_network_data, dtype=tf.float32)
        else:
            raise ValueError(f"Invalid recurrent weight regularizer type: {flags.recurrent_weight_regularizer_type}")
        loss_components['rec_weight_regularizer'] = lambda: rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values)
    else:
        loss_components['rec_weight_regularizer'] = lambda: tf.constant(0.0, dtype=dtype)

    # Rate Regularizers (Evoked and Spontaneous)
    rate_core_mask = None if flags.all_neuron_rate_loss else core_mask
    evoked_rate_regularizer = losses.SpikeRateDistributionTarget(network, spontaneous_fr=False, rate_cost=flags.rate_cost, 
                                                                pre_delay=delays[0], post_delay=delays[1],
                                                                data_dir=flags.data_dir, core_mask=rate_core_mask, seed=flags.seed, 
                                                                dtype=tf.float32, neuropixels_df=flags.neuropixels_df if hasattr(flags, 'neuropixels_df') else None)
    loss_components['evoked_rate_regularizer'] = lambda spikes: evoked_rate_regularizer(spikes)

    spont_rate_regularizer = losses.SpikeRateDistributionTarget(network, spontaneous_fr=True, rate_cost=flags.rate_cost, 
                                                               pre_delay=delays[0], post_delay=delays[1],
                                                               data_dir=flags.data_dir, core_mask=rate_core_mask, seed=flags.seed, 
                                                               dtype=tf.float32, neuropixels_df=flags.neuropixels_df if hasattr(flags, 'neuropixels_df') else None)
    loss_components['spont_rate_regularizer'] = lambda spikes: spont_rate_regularizer(spikes)

    # Voltage Regularizer
    voltage_regularizer = losses.VoltageRegularization(rsnn_layer.cell, voltage_cost=flags.voltage_cost, dtype=tf.float32) # Original used core_mask here too
    loss_components['voltage_regularizer'] = lambda voltage: voltage_regularizer(voltage)

    # OSI/DSI Loss
    # EMA for V1 firing rates
    v1_ema_path = os.path.join(logdir, 'train_end_data.pkl')
    if os.path.exists(v1_ema_path):
        with open(v1_ema_path, 'rb') as f:
            data_loaded = pkl.load(f)
            initial_v1_ema_value = data_loaded.get('v1_ema', 0.003 * np.ones(network["n_nodes"], dtype=np.float32))
    else:
        initial_v1_ema_value = 0.003 * np.ones(network["n_nodes"], dtype=np.float32) # Default: 3Hz
    
    v1_ema = tf.Variable(initial_v1_ema_value, trainable=False, name='V1_EMA', dtype=tf.float32)
    loss_components['v1_ema'] = v1_ema

    layer_info = other_v1_utils.get_layer_info(network) if flags.osi_loss_method == 'neuropixels_fr' else None
    
    osi_dsi_loss_obj = losses.OrientationSelectivityLoss(network=network, osi_cost=flags.osi_cost,
                                                         pre_delay=delays[0], post_delay=delays[1],
                                                         dtype=tf.float32, core_mask=core_mask,
                                                         method=flags.osi_loss_method,
                                                         subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                         layer_info=layer_info,
                                                         neuropixels_df=flags.neuropixels_df if hasattr(flags, 'neuropixels_df') else None)
    loss_components['osi_dsi_loss_fn'] = osi_dsi_loss_obj # pass the object

    # Annulus Regularizers (if annulus_mask exists)
    if annulus_mask is not None:
        annulus_spont_rate_regularizer = losses.SpikeRateDistributionTarget(network, spontaneous_fr=True, rate_cost=0.1*flags.rate_cost, 
                                                                            pre_delay=delays[0], post_delay=delays[1],
                                                                            data_dir=flags.data_dir, core_mask=annulus_mask, seed=flags.seed, dtype=tf.float32,
                                                                            neuropixels_df=flags.neuropixels_df if hasattr(flags, 'neuropixels_df') else None)
        loss_components['annulus_spont_rate_regularizer'] = lambda spikes: annulus_spont_rate_regularizer(spikes)

        annulus_evoked_rate_regularizer = losses.SpikeRateDistributionTarget(network, spontaneous_fr=False, rate_cost=0.1*flags.rate_cost, 
                                                                             pre_delay=delays[0], post_delay=delays[1],
                                                                             data_dir=flags.data_dir, core_mask=annulus_mask, seed=flags.seed, dtype=tf.float32,
                                                                             neuropixels_df=flags.neuropixels_df if hasattr(flags, 'neuropixels_df') else None)
        loss_components['annulus_evoked_rate_regularizer'] = lambda spikes: annulus_evoked_rate_regularizer(spikes)
        
        annulus_osi_dsi_loss_obj = losses.OrientationSelectivityLoss(network=network, osi_cost=0.1*flags.osi_cost,
                                                                     pre_delay=delays[0], post_delay=delays[1],
                                                                     dtype=tf.float32, core_mask=annulus_mask,
                                                                     method=flags.osi_loss_method,
                                                                     subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                                     layer_info=layer_info,
                                                                     neuropixels_df=flags.neuropixels_df if hasattr(flags, 'neuropixels_df') else None)
        loss_components['annulus_osi_dsi_loss_fn'] = annulus_osi_dsi_loss_obj
    else:
        loss_components['annulus_spont_rate_regularizer'] = lambda spikes: tf.constant(0.0, dtype=dtype)
        loss_components['annulus_evoked_rate_regularizer'] = lambda spikes: tf.constant(0.0, dtype=dtype)
        loss_components['annulus_osi_dsi_loss_fn'] = lambda s,a,t,n: tf.constant(0.0, dtype=dtype) # Mock function

    # Extractor Model
    extractor_model = tf.keras.Model(inputs=model.inputs, outputs=rsnn_layer.output)
    loss_components['extractor_model'] = extractor_model
    
    # Zero state (useful for train_step)
    loss_components['zero_state'] = rsnn_layer.cell.zero_state(per_replica_batch_size, dtype=dtype)

    print("Losses and regularizers setup complete.")
    return loss_components

# TODO: Add functions for train/validation steps, callback setup, and the main training loop.

if __name__ == '__main__':
    # Example usage (requires absl flags to be defined elsewhere or mocked)
    # This is for basic testing of the functions if run directly.
    
    # Mock absl.flags.FLAGS for testing
    class MockFlags:
        def __init__(self):
            self.seed = 42
            self.dtype = 'float32' # 'float16', 'bfloat16'
            self.ckpt_dir = '' # '/path/to/specific/checkpoint_run'
            self.results_dir = './tf_training_results' # Base directory for results
            self.task_name = 'test_task'
            self.neurons = 256
            self.n_input = 100 # Example, adjust as per actual flags
            self.core_only = False
            self.connected_selection = True
            self.random_weights = False
            self.uniform_weights = False
            self.optimizer = 'adam'
            self.learning_rate = 1e-3
            self.restore_from = '' # '/path/to/previous/run/Intermediate_checkpoints'
            self.batch_size = 32
            self.seq_len = 100
            self.n_output = 10
            self.cue_duration = 10
            self.input_weight_scale = 1.0
            self.dampening_factor = 0.5
            self.recurrent_dampening_factor = 0.5
            self.gauss_std = 0.1
            self.lr_scale = True
            self.train_input = True
            self.train_noise = True
            self.train_recurrent = True
            self.train_recurrent_per_type = False
            self.neuron_output = False
            self.pseudo_gauss = False
            self.hard_reset = False
            self.current_input = False
            self.max_delay = 5
            self.data_dir = 'GLIF_network'  # Updated data_dir
            self.bmtk_compat_lgn = False
            self.rotation = 0.0
            self.caching = False # For load_sparse
            # Flags for losses and regularizers
            self.loss_core_radius = 100.0
            self.recurrent_weight_regularization = 0.01
            self.uniform_weights = False # Affects rec_weight_regularizer
            self.recurrent_weight_regularizer_type = 'mean' # 'mean' or 'emd'
            self.all_neuron_rate_loss = False
            self.rate_cost = 0.1
            self.neuropixels_df = None # Path to a dataframe or None
            self.voltage_cost = 0.01
            self.osi_cost = 0.1
            self.osi_loss_method = 'gds' # 'gds', 'neuropixels_fr', etc.
            self.osi_loss_subtraction_ratio = 0.5
            self.neurons_per_output = 1 # Added based on load_sparse.py requirement


            # To access default values like in absl.flags
            self._defaults = {
                'n_input': 0, 'core_only': True, 'connected_selection': False, 
                'random_weights': True, 'uniform_weights': False,
                'results_dir': '', 'task_name': 'default_task', 'ckpt_dir': '',
                'restore_from': '', 'batch_size': 32, 'seq_len': 100, 'n_output':10,
                'cue_duration':10, 'input_weight_scale':1.0, 'dampening_factor':0.5,
                'recurrent_dampening_factor':0.5, 'gauss_std':0.1, 'lr_scale':True,
                'train_input':True, 'train_noise':True, 'train_recurrent':True,
                'train_recurrent_per_type':False, 'neuron_output':False, 'pseudo_gauss':False,
                'hard_reset':False, 'current_input':False, 'max_delay':5, 'data_dir':'GLIF_network',
                'bmtk_compat_lgn':False, 'rotation':0.0, 'caching':False,
                'loss_core_radius': 0.0, 'recurrent_weight_regularization': 0.0, 
                'recurrent_weight_regularizer_type': 'mean', 'all_neuron_rate_loss': True,
                'rate_cost': 0.0, 'neuropixels_df': None, 'voltage_cost': 0.0, 'osi_cost': 0.0,
                'osi_loss_method': 'gds', 'osi_loss_subtraction_ratio': 0.0,
                'neurons_per_output': 1
            }
            # Ensure all attributes are in _defaults for __getitem__
            for key in self.__dict__.keys():
                if key not in self._defaults and key != '_defaults':
                    self._defaults[key] = getattr(self,key) # Add with its own value as default

        def __getitem__(self, name):
            # Mocking the dictionary-like access for flags[name].default and flags[name].value
            class FlagValue:
                def __init__(self, value, default):
                    self.value = value
                    self.default = default
            return FlagValue(getattr(self, name), self._defaults.get(name))

    FLAGS = MockFlags()
    # Make sure results_dir exists for prepare_log_directory test
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir, exist_ok=True)

    # Create dummy network files for load_network_data test, using FLAGS.data_dir
    # FLAGS.data_dir is now 'GLIF_network'
    base_data_dir = FLAGS.data_dir # Should be 'GLIF_network'
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir, exist_ok=True)

    # Path for network_data.pkl: GLIF_network/v1_{neurons}/network_data.pkl
    network_specific_dir = os.path.join(base_data_dir, f'v1_{FLAGS.neurons}')
    if not os.path.exists(network_specific_dir):
        os.makedirs(network_specific_dir, exist_ok=True)
    dummy_network_path = os.path.join(network_specific_dir, 'network_data.pkl')
    if not os.path.exists(dummy_network_path):
        with open(dummy_network_path, 'wb') as f:
            mock_network_content = {
                'n_nodes': FLAGS.neurons,
                'adj': np.random.rand(FLAGS.neurons, FLAGS.neurons) > 0.9, # sparse adj
                'x': np.random.rand(FLAGS.neurons),
                'y': np.random.rand(FLAGS.neurons),
                'z': np.random.rand(FLAGS.neurons),
                'inhibitory': np.random.choice([True, False], size=FLAGS.neurons),
                'syn_type_matrix': np.random.randint(0,4, size=(FLAGS.neurons, FLAGS.neurons))
            }
            pkl.dump(mock_network_content, f)

    # Path for v1_node_types.csv: GLIF_network/network/v1_node_types.csv
    dummy_node_types_structural_dir = os.path.join(base_data_dir, 'network')
    if not os.path.exists(dummy_node_types_structural_dir):
        os.makedirs(dummy_node_types_structural_dir, exist_ok=True)
    dummy_node_types_path = os.path.join(dummy_node_types_structural_dir, 'v1_node_types.csv')
    if not os.path.exists(dummy_node_types_path):
        with open(dummy_node_types_path, 'w') as f:
            f.write("node_type_id pop_name\n")
            f.write("0 Excitatory\n")
            f.write("1 Inhibitory\n")

    # Dummy load_fn for testing setup_losses_and_regularizers (uses the pkl created above)
    def mock_load_fn(flags_obj, neurons_val, flag_str_val):
        # This mock_load_fn is for the setup_losses_and_regularizers internal call if flags.uniform_weights is true
        # It should load the network_data.pkl we created
        path_to_load = os.path.join(flags_obj.data_dir, f'v1_{neurons_val}', 'network_data.pkl')
        with open(path_to_load, 'rb') as f:
            net = pkl.load(f)
        return net, None, None # network, lgn_input, bkg_input

    print("--- Testing training_utils.py ---")
    setup_environment(seed=FLAGS.seed)
    dtype = configure_mixed_precision(FLAGS.dtype)
    strategy = create_distribution_strategy()
    logdir, flag_str_log = prepare_log_directory(FLAGS)
    print(f"Log directory: {logdir}, Flag string: {flag_str_log}")

    network, lgn_input, bkg_input = load_network_data(FLAGS, flag_str_log)
    print(f"Network loaded: {network is not None} with {network['n_nodes']} nodes")
    delays_example = [200, 200] # Example delays

    with strategy.scope():
        per_replica_bs = FLAGS.batch_size // strategy.num_replicas_in_sync
        model = create_v1_model(network, lgn_input, bkg_input, FLAGS, dtype, per_replica_bs)
        model.build((per_replica_bs, FLAGS.seq_len, FLAGS.n_input))
        print(f"V1 Model created and built.")
        
        optimizer = create_optimizer(FLAGS.optimizer, FLAGS.learning_rate, FLAGS.dtype)
        optimizer.build(model.trainable_variables)
        print(f"Optimizer created: {optimizer.name}")

        restored, optimizer = restore_checkpoint(model, optimizer, FLAGS)
        print(f"Checkpoint restored: {restored}, Optimizer after restore: {optimizer.name}")
        
        # Test dataset functions
        gratings_fn = get_gratings_dataset_fn(FLAGS, FLAGS.batch_size, dtype, delays_example)
        gray_fn = get_gray_dataset_fn(FLAGS, FLAGS.batch_size, dtype)

        dist_gratings_dataset = strategy.distribute_datasets_from_function(gratings_fn)
        dist_gray_dataset = strategy.distribute_datasets_from_function(gray_fn)

        print(f"Gratins dataset element spec: {dist_gratings_dataset.element_spec}")
        print(f"Gray screen dataset element spec: {dist_gray_dataset.element_spec}")
        # for x_batch, y_batch in dist_gratings_dataset.take(1):
        #     print(f"Sample gratings batch X shape: {x_batch.shape}, Y shape: {y_batch.shape}")
        # for x_batch_gray, y_batch_gray in dist_gray_dataset.take(1):
        #     print(f"Sample gray batch X shape: {x_batch_gray.shape}, Y shape: {y_batch_gray.shape}")
        
        # Test setup_losses_and_regularizers
        loss_components = setup_losses_and_regularizers(FLAGS, model, network, mock_load_fn, dtype, per_replica_bs, logdir, delays_example, flag_str_log)
        print(f"Loss components created. Keys: {list(loss_components.keys())}")
        # Example: Check if extractor model output matches rsnn_layer output spec
        dummy_input = tf.random.normal((per_replica_bs, FLAGS.seq_len, FLAGS.n_input), dtype=dtype)
        rsnn_output_spec = model.get_layer('rsnn').compute_output_signature(tf.TensorSpec(shape=(per_replica_bs, FLAGS.seq_len, FLAGS.n_input), dtype=dtype))
        extractor_output = loss_components['extractor_model'](dummy_input)
        tf.nest.map_structure(lambda spec, out: tf.debugging.assert_near(tf.zeros_like(out), out, atol=1e5, message=f"Spec: {spec.shape}, Output: {out.shape}"), rsnn_output_spec, extractor_output) # Check shape and type by comparing to spec
        print("Extractor model output spec matches rsnn layer.")

    print("--- Testing completed ---")
    # Clean up dummy files/dirs if necessary
    # Cleanup
    if os.path.exists(dummy_network_path): # GLIF_network/v1_256/network_data.pkl
        os.remove(dummy_network_path)
    if os.path.exists(dummy_node_types_path): # GLIF_network/network/v1_node_types.csv
        os.remove(dummy_node_types_path)
    
    # Clean up directories if they are empty
    if os.path.exists(dummy_node_types_structural_dir) and not os.listdir(dummy_node_types_structural_dir):
        os.rmdir(dummy_node_types_structural_dir) # GLIF_network/network/
    if os.path.exists(network_specific_dir) and not os.listdir(network_specific_dir):
        os.rmdir(network_specific_dir) # GLIF_network/v1_256/
    if os.path.exists(FLAGS.data_dir) and not os.listdir(FLAGS.data_dir):
        os.rmdir(FLAGS.data_dir) # GLIF_network/

    # Cleanup for tf_data directory created by load_sparse.create_network_dat
    tf_data_dir = os.path.join(FLAGS.data_dir, 'tf_data')
    # The actual file is network_dat.pkl inside tf_data
    network_dat_pkl_path = os.path.join(tf_data_dir, 'network_dat.pkl') 
    if os.path.exists(network_dat_pkl_path):
        os.remove(network_dat_pkl_path)
    if os.path.exists(tf_data_dir) and not os.listdir(tf_data_dir):
        os.rmdir(tf_data_dir)

    # import shutil
    # if os.path.exists(FLAGS.results_dir):
    #     shutil.rmtree(FLAGS.results_dir)
    # if os.path.exists(logdir) and logdir != FLAGS.results_dir : # avoid deleting results_dir if it's the same
    #    shutil.rmtree(logdir)
