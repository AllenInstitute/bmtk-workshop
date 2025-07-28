import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def gaussian_kernel(sigma, kernel_size=100):
    """Create Gaussian kernel for smoothing"""
    if kernel_size is None:
        kernel_size = int(4 * sigma) + 1
    x = np.arange(kernel_size) - (kernel_size - 1) / 2
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / np.sum(kernel)

def detect_spikes_threshold(voltage, threshold=-20.0, dt=0.1):
    """
    Simple threshold-based spike detection.
    
    Parameters:
    - voltage: array of voltage values (mV)
    - threshold: spike threshold (mV)
    - dt: time step (ms)
    
    Returns:
    - spike_times: array of spike times
    - spike_indices: array of spike indices
    - spike_train: binary array (1 at spike indices, 0 elsewhere)
    """
    # Find where voltage crosses threshold upward
    spike_indices, _ = find_peaks(np.squeeze(voltage), height=threshold, distance=1)
    spike_times = spike_indices * dt
    
    # Create binary spike train
    spike_train = np.zeros_like(voltage)
    if len(spike_indices) > 0:
        spike_train[spike_indices] = 1
    
    return spike_times, spike_indices, spike_train

# Alternative implementation using scipy's gaussian_filter1d for better performance
def smooth_firing_rate_scipy(v, threshold, dt, sigma=5.0):
    """
    Convert voltage to continuous firing rate with Gaussian smoothing
    """
    # detect spikes
    spike_times, spike_indices, spike_train = detect_spikes_threshold(v, threshold=threshold, dt=dt)
    
    # Apply Gaussian smoothing using scipy
    rate = gaussian_filter1d(spike_train, sigma=sigma, axis=0)
    
    return rate, spike_times, spike_indices, spike_train

def calc_rate_loss(test_vData, target_vData, threshold, dt, sigma):
    
    target_rate, _, _, _ = smooth_firing_rate_scipy(target_vData, threshold, dt, sigma=sigma)
    test_rate, _, _, _= smooth_firing_rate_scipy(test_vData, threshold, dt, sigma=sigma)

    rate_loss = np.mean(np.sqrt((test_rate - target_rate)**2))
    rate_loss = rate_loss * 1e5

    return rate_loss, target_rate, test_rate
