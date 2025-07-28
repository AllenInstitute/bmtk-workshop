import os
import numpy as np
import tensorflow as tf
from packaging import version
from v1_model_utils import load_sparse, models, other_v1_utils
from v1_model_utils.loss_functions import *
from v1_model_utils.callbacks import Callbacks
from optimizers import ExponentiatedAdam

# 1. Prepare environment (seeds, dtype, mixed precision)
def prepare_environment(seed=0, dtype='float32'):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)
    if dtype == 'float16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        else:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
        tf_dtype = tf.float16
    elif dtype == 'bfloat16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy("mixed_bfloat16")
            mixed_precision.set_policy(policy)
        else:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_bfloat16')
        tf_dtype = tf.bfloat16
    else:
        tf_dtype = tf.float32
    return tf_dtype

# 2. Load network, lgn_input, bkg_input
def load_v1_network(flags, flag_str=None):
    if getattr(flags, 'caching', False):
        load_fn = load_sparse.cached_load_v1
    else:
        load_fn = load_sparse.load_v1
    return load_fn(flags, flags.neurons, flag_str=flag_str)

# 3. Build model
def build_v1_model(flags, network, lgn_input, bkg_input, dtype=tf.float32):
    return models.create_model(
        network,
        lgn_input,
        bkg_input,
        seq_len=flags.seq_len,
        n_input=flags.n_input,
        n_output=flags.n_output,
        cue_duration=getattr(flags, 'cue_duration', 20),
        dtype=dtype,
        batch_size=flags.batch_size,
        input_weight_scale=getattr(flags, 'input_weight_scale', 1.0),
        dampening_factor=getattr(flags, 'dampening_factor', 0.2),
        recurrent_dampening_factor=getattr(flags, 'recurrent_dampening_factor', 0.5),
        gauss_std=getattr(flags, 'gauss_std', 0.5),
        lr_scale=getattr(flags, 'lr_scale', 800.0),
        train_input=getattr(flags, 'train_input', True),
        train_noise=getattr(flags, 'train_noise', True),
        train_recurrent=getattr(flags, 'train_recurrent', True),
        train_recurrent_per_type=getattr(flags, 'train_recurrent_per_type', False),
        neuron_output=getattr(flags, 'neuron_output', False),
        pseudo_gauss=getattr(flags, 'pseudo_gauss', False),
        use_state_input=True,
        return_state=True,
        hard_reset=getattr(flags, 'hard_reset', False),
        add_metric=False,
        max_delay=5,
        current_input=getattr(flags, 'current_input', False)
    )

# 4. Get optimizer
def get_optimizer(flags, model, dtype=tf.float32):
    if flags.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)
    elif flags.optimizer == 'exp_adam':
        optimizer = ExponentiatedAdam(flags.learning_rate, epsilon=1e-11)
    elif flags.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(flags.learning_rate, momentum=0.0, nesterov=False)
    else:
        raise ValueError(f"Invalid optimizer: {flags.optimizer}")
    optimizer.build(model.trainable_variables)
    if dtype == tf.float16:
        from tensorflow.keras import mixed_precision
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    return optimizer

# 5. Example: training loop stub (to be expanded as needed)
def train_v1_model(model, optimizer, train_dataset, flags, callbacks, strategy=None):
    # This is a stub. For a full training loop, adapt from main().
    # For notebook demo, you can run one epoch as a proof of concept.
    for batch in train_dataset:
        with tf.GradientTape() as tape:
            outputs = model(batch[0], training=True)
            loss = ... # define loss here
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # callbacks, metrics, etc.
        break # For notebook demo, just run one batch
    print("Training step complete (demo)")

# Add more utilities as needed for notebook use.
