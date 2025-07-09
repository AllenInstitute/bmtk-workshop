## Setup and Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from time import time

# Import our tutorial support functions
import v1_tutorial_support as vts
from importlib import reload
reload(vts)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")


## Configuration
# Create configuration object
config = vts.TutorialConfig()

# Tutorial settings (smaller scale for demonstration)
config.n_epochs = 10
config.steps_per_epoch = 10
config.neurons = 5000  # Use subset for faster demo

print("Training Configuration:")
print(f"  Epochs: {config.n_epochs}")
print(f"  Steps per epoch: {config.steps_per_epoch}")
print(f"  Neurons: {config.neurons} (0 = all)")
print(f"  Learning rate: {config.learning_rate}")


## Environment Setup
# Setup environment
vts.setup_environment(config)
dtype = vts.configure_mixed_precision(config.dtype)
strategy = vts.create_distribution_strategy()
logdir, flag_str = vts.prepare_directories(config)

print(f"Results will be saved to: {logdir}")

# Calculate batch sizes
per_replica_batch_size = config.batch_size
global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
print(f"Batch sizes: {per_replica_batch_size} per replica, {global_batch_size} global")


## Load Network
# Load network data
network, lgn_input, bkg_input = vts.load_network_data(config, flag_str)
delays = [200, 0]  # [pre_delay, post_delay] in ms

print(f"Network loaded with {network['n_nodes']} neurons")
print(f"Stimulus timing: pre={delays[0]}ms, post={delays[1]}ms")

## Create Model and Optimizer
# Create model within strategy scope
with strategy.scope():
    model = vts.create_model(
        config, network, lgn_input, bkg_input, 
        dtype, per_replica_batch_size
    )
    
    optimizer = vts.create_optimizer(config, model, config.dtype)
    
    print(f"Model created with {model.count_params():,} parameters")
    print(f"Optimizer: {type(optimizer).__name__}")
    
    # Store initial weights for comparison
    initial_weights = [var.numpy().copy() for var in model.trainable_variables[:2]]


## Setup Loss Functions
with strategy.scope():
    # Setup spatial masks and loss functions
    core_mask, annulus_mask = vts.setup_core_masks(config, network)
    loss_components = vts.create_loss_functions(
        config, model, network, dtype, delays, core_mask, logdir
    )
    metrics, reset_metrics = vts.create_training_metrics()
    
    print("Loss functions configured:")
    print("✓ Rate distribution matching")
    print("✓ Voltage regularization") 


## Setup Input Data Generation by LGN
# Create dataset functions
get_gratings_dataset_fn, _ = vts.create_dataset_functions(
    config, global_batch_size, dtype, delays
)
train_data_set = strategy.distribute_datasets_from_function(get_gratings_dataset_fn())

# Precompute spontaneous firing rates
spontaneous_prob = vts.compute_spontaneous_lgn_rates(config, dtype)
print(f"Data pipeline ready, spontaneous rates: {spontaneous_prob.shape}")


## Define Training Functions
@tf.function()  # accelerate training by using tf.function
def simple_training_step(x, y):
    """Simplified training step for tutorial"""
    with tf.GradientTape() as tape:
        # Forward pass through model
        rsnn_layer = loss_components['rsnn_layer']
        dummy_zeros = tf.zeros((per_replica_batch_size, tf.shape(x)[1], network["n_nodes"]), dtype)
        zero_state = rsnn_layer.cell.zero_state(per_replica_batch_size, dtype)
        
        extractor_model = tf.keras.Model(inputs=model.inputs, outputs=rsnn_layer.output)
        out = extractor_model((x, dummy_zeros, zero_state))
        z, v = out[0]  # spikes, voltages
        
        # Compute losses (simplified for tutorial)
        rate_loss = loss_components['evoked_rate_regularizer'](z, True)
        voltage_loss = loss_components['voltage_regularizer'](v)
        
        total_loss = rate_loss + voltage_loss
    
        scaled_loss = total_loss
    
    # Compute and apply gradients
    gradients = tape.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, z, rate_loss, voltage_loss

print("Training functions ready")


## Training Loop
# Initialize training history
training_history = {
    'epochs': [], 'loss': [], 'firing_rate': [], 
    'rate_loss': [], 'voltage_loss': []
}

print("Starting training...")
print("-" * 50)

for epoch in range(config.n_epochs):
    epoch_start = time()
    epoch_losses, epoch_rates = [], []
    epoch_rate_losses, epoch_voltage_losses, epoch_osi_losses = [], [], []
    
    # Create fresh dataset iterator
    train_iterator = iter(train_data_set)
    
    for step in range(config.steps_per_epoch):
        # Get training batch
        x, y, _, _ = next(train_iterator)
        
        # Extract from distribution strategy if needed
        if strategy.num_replicas_in_sync > 1:
            x = strategy.experimental_local_results(x)[0]
            y = strategy.experimental_local_results(y)[0]
        
        # Training step
        loss, spikes, rate_loss, voltage_loss = simple_training_step(x, y)
        
        # Record metrics
        epoch_losses.append(loss.numpy())
        firing_rate = tf.reduce_mean(spikes).numpy() * 1000  # Convert to Hz
        epoch_rates.append(firing_rate)
        epoch_rate_losses.append(rate_loss.numpy())
        epoch_voltage_losses.append(voltage_loss.numpy())
        
        if step % 2 == 0:
            print(f"  Step {step+1}: Loss={loss.numpy():.4f}, Rate={firing_rate:.2f} Hz")
    
    # Epoch summary
    epoch_time = time() - epoch_start
    avg_loss = np.mean(epoch_losses)
    avg_rate = np.mean(epoch_rates)
    
    # Store history
    training_history['epochs'].append(epoch + 1)
    training_history['loss'].append(avg_loss)
    training_history['firing_rate'].append(avg_rate)
    training_history['rate_loss'].append(np.mean(epoch_rate_losses))
    training_history['voltage_loss'].append(np.mean(epoch_voltage_losses))
    
    print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Rate={avg_rate:.2f} Hz, Time={epoch_time:.1f}s")
    print("-" * 50)

print("Training completed!")


## Visualize Training Progress
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Training Progress', fontsize=16)

# Total loss
axes[0, 0].plot(training_history['epochs'], training_history['loss'], 'b-o', linewidth=2)
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Firing rate
axes[0, 1].plot(training_history['epochs'], training_history['firing_rate'], 'g-o', linewidth=2)
axes[0, 1].set_title('Average Firing Rate')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Rate (Hz)')
axes[0, 1].grid(True, alpha=0.3)

# Rate loss
axes[1, 0].plot(training_history['epochs'], training_history['rate_loss'], 'r-o', linewidth=2)
axes[1, 0].set_title('Rate Distribution Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Rate Loss')
axes[1, 0].grid(True, alpha=0.3)

# OSI loss
axes[1, 1].plot(training_history['epochs'], training_history['voltage_loss'], 'c-o', linewidth=2)
axes[1, 1].set_title('Voltage Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Voltage Loss')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Loss improved from {training_history['loss'][0]:.4f} to {training_history['loss'][-1]:.4f}")


## Analyze Network Activity
# Test network response
test_iterator = iter(train_data_set)
x_test, y_test, _, _ = next(test_iterator)

if strategy.num_replicas_in_sync > 1:
    x_test = strategy.experimental_local_results(x_test)[0]
    y_test = strategy.experimental_local_results(y_test)[0]

# Get network response
_, spikes, _, _ = simple_training_step(x_test, y_test)
spikes_np = spikes.numpy()[0]  # First batch item

# Analyze activity
pop_rate = np.mean(spikes_np, axis=1) * 1000  # Convert to Hz
mean_rate = np.mean(pop_rate)
stim_start, stim_end = delays[0], len(pop_rate) - delays[1]
stim_rate = np.mean(pop_rate[stim_start:stim_end])
baseline_rate = np.mean(np.concatenate([pop_rate[:stim_start], pop_rate[stim_end:]]))

print(f"Network Activity:")
print(f"  Average firing rate: {mean_rate:.2f} Hz")
print(f"  Stimulus period firing rate: {stim_rate:.2f} Hz")
print(f"  Spontaneous period firing rate: {baseline_rate:.2f} Hz")

max(pop_rate)


## Visualize Network Response
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Spike raster (subset of neurons)
n_neurons_plot = min(20000, spikes_np.shape[1])

# pick neurons from masked region

time_ms = np.arange(spikes_np.shape[0])
# spike_times, spike_neurons = np.where(spikes_np[:, :n_neurons_plot] > 0.5)
spike_times, spike_neurons = np.where(spikes_np > 0.5)

ax1.scatter(spike_times, spike_neurons, s=1, alpha=0.7, c='red')
ax1.set_title(f'Spike Raster ({n_neurons_plot} neurons)')
ax1.set_ylabel('Neuron Index')
ax1.axvline(delays[0], color='green', linestyle='--', alpha=0.7, label='Stimulus ON')
ax1.axvline(len(time_ms) - delays[1], color='red', linestyle='--', alpha=0.7, label='Stimulus OFF')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Population firing rate
ax2.plot(time_ms, pop_rate, 'blue', linewidth=1.5)
ax2.set_title('Population Firing Rate')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Rate (Hz)')
ax2.axvline(delays[0], color='green', linestyle='--', alpha=0.7, label='Stimulus ON')
ax2.axvline(len(time_ms) - delays[1], color='red', linestyle='--', alpha=0.7, label='Stimulus OFF')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


## Weight Change Analysis
# Analyze weight changes
final_weights = [var.numpy() for var in model.trainable_variables[:2]]
weight_changes = []

for initial, final in zip(initial_weights, final_weights):
    change = np.mean(np.abs(final - initial) / (np.abs(initial) + 1e-8))
    weight_changes.append(change)

layer_names = ["Recurrent", "Background"]

print("Weight Change Analysis:")
for i, change in enumerate(weight_changes):
    print(f"  {layer_names[i]}: {change:.4f} relative change")

# Plot weight change distribution
if len(weight_changes) > 0:
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(weight_changes)), weight_changes)
    plt.title('Weight Changes by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Relative Change')
    plt.xticks(range(len(weight_changes)), [f'{layer_names[i]}' for i in range(len(weight_changes))])
    plt.grid(True, alpha=0.3)
    plt.show()


## Save Results
# Save training history
import pickle
with open(os.path.join(logdir, 'training_history.pkl'), 'wb') as f:
    pickle.dump(training_history, f)

# Save configuration
with open(os.path.join(logdir, 'config.txt'), 'w') as f:
    f.write("Tutorial Configuration:\n")
    for attr in dir(config):
        if not attr.startswith('_'):
            f.write(f"{attr}: {getattr(config, attr)}\n")

print(f"Results saved to: {logdir}")


