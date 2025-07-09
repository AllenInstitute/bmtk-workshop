"""
Support functions for V1 Model Training Tutorial

This module contains clean, extracted functions from multi_training_single_gpu_split.py
to support tutorial notebook usage without breaking existing library functionality.
"""

import os
import copy
import random
import numpy as np
import tensorflow as tf
import pickle as pkl
from packaging import version
from time import time

# Import existing library functions (don't modify these)
from v1_model_utils import load_sparse, models, other_v1_utils, toolkit
import v1_model_utils.loss_functions as losses
from v1_model_utils.callbacks import Callbacks
import stim_dataset
from optimizers import ExponentiatedAdam


class TutorialConfig:
    """Simple configuration class for tutorial settings"""
    def __init__(self):
        # Basic settings
        self.seed = 3000
        self.dtype = 'float32'  # 'float16', 'bfloat16', 'float32'
        self.data_dir = 'GLIF_network'
        self.results_dir = 'tutorial_results'
        self.task_name = 'v1_tutorial'
        
        # Model settings
        self.neurons = 5000  # 0 means use all neurons
        self.n_input = 696
        self.batch_size = 10
        self.seq_len = 500
        self.n_output = 2
        self.cue_duration = 40
        
        # Training settings
        self.learning_rate = 0.005
        self.optimizer = 'exp_adam'
        self.n_epochs = 10
        self.steps_per_epoch = 20
        
        # Model parameters
        self.input_weight_scale = 1.8
        self.dampening_factor = 0.5
        self.recurrent_dampening_factor = 0.5
        self.gauss_std = 0.3
        self.lr_scale = 1.0
        self.train_input = False
        self.train_noise = True
        self.train_recurrent = True
        self.train_recurrent_per_type = False
        self.neuron_output = False
        self.neurons_per_output = 16
        self.pseudo_gauss = False
        self.hard_reset = False
        self.current_input = False
        
        # Loss and regularization
        self.rate_cost = 10000.0
        self.voltage_cost = 1.0
        self.osi_cost = 20.0
        self.sync_cost = 1.5
        self.recurrent_weight_regularization = 10.0
        self.recurrent_weight_regularizer_type = 'emd'
        self.loss_core_radius = 200.0
        self.all_neuron_rate_loss = False
        self.osi_loss_method = 'crowd_osi'
        self.osi_loss_subtraction_ratio = 1.0
        
        # Dataset settings
        self.rotation = 'ccw'
        self.bmtk_compat_lgn = True
        self.caching = True
        self.core_only = False
        self.connected_selection = True
        self.random_weights = False
        self.uniform_weights = False
        
        # Files
        self.neuropixels_df = 'Neuropixels_data/OSI_DSI_neuropixels_v4.csv'
        self.fano_samples = 500


def setup_environment(config):
    """Setup environment for reproducible training"""
    # Set environment variables for optimal GPU performance
    os.environ['TF_GPU_THREAD_MODE'] = 'global'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # Set seeds for reproducibility
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    random.seed(config.seed)
    
    # Allow GPU memory growth
    physical_devices = tf.config.list_physical_devices("GPU")
    for dev in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(dev, True)
            print(f"Memory growth enabled for device {dev}")
        except:
            print(f"Invalid device {dev} or cannot modify virtual devices once initialized.")
    
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"TensorFlow version: {tf.__version__}")
    

def configure_mixed_precision(dtype_str):
    """Configure mixed precision training"""
    if dtype_str == 'float16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        else:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
        dtype = tf.float16
        print('Mixed precision (float16) enabled!')
    elif dtype_str == 'bfloat16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy("mixed_bfloat16")
            mixed_precision.set_policy(policy)
        else:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_bfloat16')
        dtype = tf.bfloat16
        print('Mixed precision (bfloat16) enabled!')
    else:
        dtype = tf.float32
        print('Using float32 precision')
    
    return dtype


def create_distribution_strategy():
    """Create TensorFlow distribution strategy"""
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
    return strategy


def prepare_directories(config):
    """Prepare result directories"""
    # Create flag string for directory naming
    flag_str = f'v1_{config.neurons}'
    for attr in ['n_input', 'core_only', 'connected_selection', 'random_weights']:
        default_val = 0 if attr == 'n_input' else False
        if getattr(config, attr) != default_val:
            flag_str += f'_{attr}_{getattr(config, attr)}'
    
    # Create results directory
    results_dir = os.path.join(config.results_dir, flag_str)
    os.makedirs(results_dir, exist_ok=True)
    print(f'Results will be stored in: {results_dir}')
    
    # Generate unique simulation name
    sim_name = toolkit.get_random_identifier('tutorial_')
    logdir = os.path.join(results_dir, sim_name)
    os.makedirs(logdir, exist_ok=True)
    
    return logdir, flag_str


def load_network_data(config, flag_str):
    """Load network architecture and input data"""
    print("Loading network data...")
    t0 = time()
    
    # Choose caching based on config
    if config.caching:
        load_fn = load_sparse.cached_load_v1
    else:
        load_fn = load_sparse.load_v1
    
    network, lgn_input, bkg_input = load_fn(config, config.neurons, flag_str=flag_str)
    print(f"Model files loading: {time()-t0:.2f} seconds")
    print(f"Network loaded with {network['n_nodes']} neurons")
    
    return network, lgn_input, bkg_input


def create_model(config, network, lgn_input, bkg_input, dtype, batch_size):
    """Create the V1 model"""
    print("Creating model...")
    t0 = time()
    
    model = models.create_model(
        network=network,
        lgn_input=lgn_input,
        bkg_input=bkg_input,
        seq_len=config.seq_len,
        n_input=config.n_input,
        n_output=config.n_output,
        cue_duration=config.cue_duration,
        dtype=dtype,
        batch_size=batch_size,
        input_weight_scale=config.input_weight_scale,
        dampening_factor=config.dampening_factor,
        recurrent_dampening_factor=config.recurrent_dampening_factor,
        gauss_std=config.gauss_std,
        lr_scale=config.lr_scale,
        train_input=config.train_input,
        train_noise=config.train_noise,
        train_recurrent=config.train_recurrent,
        train_recurrent_per_type=config.train_recurrent_per_type,
        neuron_output=config.neuron_output,
        pseudo_gauss=config.pseudo_gauss,
        use_state_input=True,
        return_state=True,
        hard_reset=config.hard_reset,
        add_metric=False,
        max_delay=5,
        current_input=config.current_input
    )
    
    # Build model
    model.build((batch_size, config.seq_len, config.n_input))
    print(f"Model built in {time()-t0:.2f} seconds")
    
    return model


def create_optimizer(config, model, dtype_str):
    """Create and configure optimizer"""
    if config.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(config.learning_rate, epsilon=1e-11)
    elif config.optimizer == 'exp_adam':
        optimizer = ExponentiatedAdam(config.learning_rate, epsilon=1e-11)
    elif config.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(config.learning_rate, momentum=0.0, nesterov=False)
    else:
        raise ValueError(f"Invalid optimizer: {config.optimizer}")
    
    optimizer.build(model.trainable_variables)
    
    # Enable loss scaling for float16
    if dtype_str == 'float16':
        from tensorflow.keras import mixed_precision
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        print("LossScaleOptimizer enabled for float16 training")
    
    return optimizer


def setup_core_masks(config, network):
    """Setup core and annulus masks for spatial loss weighting"""
    if config.loss_core_radius > 0:
        core_mask = other_v1_utils.isolate_core_neurons(
            network, radius=config.loss_core_radius, data_dir=config.data_dir
        )
        
        if core_mask.all():
            core_mask = None
            annulus_mask = None
            print("All neurons are in the core region. Core mask is set to None.")
        else:
            print(f"Core mask includes {core_mask.sum()} neurons.")
            core_mask = tf.constant(core_mask, dtype=tf.bool)
            annulus_mask = tf.constant(~core_mask, dtype=tf.bool)
    else:
        core_mask = None
        annulus_mask = None
        print("No spatial masking applied (loss_core_radius = 0)")
    
    return core_mask, annulus_mask


def create_loss_functions(config, model, network, dtype, delays, core_mask, logdir):
    """Create all loss and regularization functions"""
    print("Setting up loss functions...")
    
    # Get RSNN layer for accessing internal states
    rsnn_layer = model.get_layer("rsnn")
    
    # Setup recurrent weight regularizer
    if config.recurrent_weight_regularization > 0 and config.uniform_weights:
        print("Loading network with original weights for regularizer...")
        dummy_config = copy.deepcopy(config)
        dummy_config.uniform_weights = False
        load_fn = load_sparse.cached_load_v1 if config.caching else load_sparse.load_v1
        rec_reg_network, _, _ = load_fn(dummy_config, dummy_config.neurons, flag_str='')
    else:
        rec_reg_network = network
    
    if config.recurrent_weight_regularizer_type == 'mean':
        rec_weight_regularizer = losses.MeanStiffRegularizer(
            config.recurrent_weight_regularization, rec_reg_network,
            penalize_relative_change=True, dtype=tf.float32
        )
    elif config.recurrent_weight_regularizer_type == 'emd':
        rec_weight_regularizer = losses.EarthMoversDistanceRegularizer(
            config.recurrent_weight_regularization, rec_reg_network, dtype=tf.float32
        )
    else:
        raise ValueError(f"Invalid regularizer type: {config.recurrent_weight_regularizer_type}")
    
    # Setup rate regularizers
    rate_core_mask = None if config.all_neuron_rate_loss else core_mask
    
    evoked_rate_regularizer = losses.SpikeRateDistributionTarget(
        network, spontaneous_fr=False, rate_cost=config.rate_cost,
        pre_delay=delays[0], post_delay=delays[1], data_dir=config.data_dir,
        core_mask=rate_core_mask, seed=config.seed, dtype=tf.float32,
        neuropixels_df=config.neuropixels_df
    )
    
    spont_rate_regularizer = losses.SpikeRateDistributionTarget(
        network, spontaneous_fr=True, rate_cost=config.rate_cost,
        pre_delay=delays[0], post_delay=delays[1], data_dir=config.data_dir,
        core_mask=rate_core_mask, seed=config.seed, dtype=tf.float32,
        neuropixels_df=config.neuropixels_df
    )
    
    # Setup voltage regularizer
    voltage_regularizer = losses.VoltageRegularization(
        rsnn_layer.cell, voltage_cost=config.voltage_cost, dtype=tf.float32
    )
    
    # Setup synchronization losses
    evoked_sync_loss = losses.SynchronizationLoss(
        network, sync_cost=config.sync_cost, core_mask=core_mask,
        t_start=0.2, t_end=config.seq_len/1000, n_samples=config.fano_samples,
        dtype=tf.float32, session='evoked', data_dir='Synchronization_data'
    )
    
    spont_sync_loss = losses.SynchronizationLoss(
        network, sync_cost=config.sync_cost, core_mask=core_mask,
        t_start=0.2, t_end=config.seq_len/1000, n_samples=config.fano_samples,
        dtype=tf.float32, session='spont', data_dir='Synchronization_data'
    )
    
    # Setup OSI/DSI loss
    layer_info = other_v1_utils.get_layer_info(network) if config.osi_loss_method == 'neuropixels_fr' else None
    
    osi_dsi_loss = losses.OrientationSelectivityLoss(
        network=network, osi_cost=config.osi_cost,
        pre_delay=delays[0], post_delay=delays[1], dtype=tf.float32,
        core_mask=core_mask, method=config.osi_loss_method,
        subtraction_ratio=config.osi_loss_subtraction_ratio,
        layer_info=layer_info, neuropixels_df=config.neuropixels_df
    )
    
    # Initialize EMA for firing rates
    if os.path.exists(os.path.join(logdir, 'train_end_data.pkl')):
        with open(os.path.join(logdir, 'train_end_data.pkl'), 'rb') as f:
            data_loaded = pkl.load(f)
            v1_ema = tf.Variable(data_loaded['v1_ema'], trainable=False, name='V1_EMA')
    else:
        v1_ema = tf.Variable(
            tf.constant(0.003, shape=(network["n_nodes"],), dtype=tf.float32),
            trainable=False, name='V1_EMA'
        )
    
    loss_components = {
        'rec_weight_regularizer': rec_weight_regularizer,
        'evoked_rate_regularizer': evoked_rate_regularizer,
        'spont_rate_regularizer': spont_rate_regularizer,
        'voltage_regularizer': voltage_regularizer,
        'evoked_sync_loss': evoked_sync_loss,
        'spont_sync_loss': spont_sync_loss,
        'osi_dsi_loss': osi_dsi_loss,
        'v1_ema': v1_ema,
        'rsnn_layer': rsnn_layer
    }
    
    print("Loss functions setup complete")
    return loss_components


def create_dataset_functions(config, global_batch_size, dtype, delays):
    """Create dataset generation functions"""
    
    def get_gratings_dataset_fn(regular=False):
        def _f(input_context):
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            return (stim_dataset.generate_drifting_grating_tuning(
                seq_len=config.seq_len,
                pre_delay=delays[0],
                post_delay=delays[1],
                n_input=config.n_input,
                data_dir=config.data_dir,
                regular=regular,
                bmtk_compat=config.bmtk_compat_lgn,
                rotation=config.rotation,
                dtype=dtype
            )
            .shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
        return _f
    
    def get_gray_dataset_fn():
        def _f(input_context):
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            return (stim_dataset.generate_drifting_grating_tuning(
                seq_len=config.seq_len,
                pre_delay=config.seq_len,
                post_delay=0,
                n_input=config.n_input,
                data_dir=config.data_dir,
                rotation=config.rotation,
                return_firing_rates=True,
                dtype=dtype
            )
            .shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
        return _f
    
    return get_gratings_dataset_fn, get_gray_dataset_fn


def compute_spontaneous_lgn_rates(config, dtype):
    """Precompute spontaneous LGN firing rates for efficiency"""
    cache_dir = f"{config.data_dir}/tf_data"
    cache_file = os.path.join(
        cache_dir, 
        f"spontaneous_lgn_probabilities_n_input_{config.n_input}_seqlen_{config.seq_len}.pkl"
    )
    
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            spontaneous_prob = pkl.load(f)
        print("Loaded cached spontaneous LGN firing rates.")
    else:
        print("Computing spontaneous LGN firing rates...")
        # Compute and cache the spontaneous firing rates
        spontaneous_lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
            seq_len=config.seq_len,
            pre_delay=config.seq_len,
            post_delay=0,
            n_input=config.n_input,
            rotation=config.rotation,
            data_dir=config.data_dir,
            billeh_phase=True,
            return_firing_rates=True,
            dtype=dtype
        )
        spontaneous_lgn_firing_rates = next(iter(spontaneous_lgn_firing_rates))
        spontaneous_prob = 1 - tf.exp(-tf.cast(spontaneous_lgn_firing_rates, dtype) / 1000.)
        
        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pkl.dump(spontaneous_prob, f)
        print("Computed and cached spontaneous LGN firing rates.")
    
    # Expand to match batch size
    spontaneous_prob = tf.tile(
        tf.expand_dims(spontaneous_prob, axis=0), 
        [config.batch_size, 1, 1]
    )
    
    return tf.cast(spontaneous_prob, dtype=dtype)


def create_training_metrics():
    """Create training metrics for monitoring"""
    metrics = {
        'train_loss': tf.keras.metrics.Mean(),
        'train_firing_rate': tf.keras.metrics.Mean(),
        'train_rate_loss': tf.keras.metrics.Mean(),
        'train_voltage_loss': tf.keras.metrics.Mean(),
        'train_regularizer_loss': tf.keras.metrics.Mean(),
        'train_osi_dsi_loss': tf.keras.metrics.Mean(),
        'train_sync_loss': tf.keras.metrics.Mean(),
    }
    
    def reset_metrics():
        for metric in metrics.values():
            metric.reset_states()
    
    return metrics, reset_metrics 