"""
Preprocessing pipeline for NinaPro EMG data.

The pipeline consists of:
1. Bandpass filtering: removes noise outside the EMG frequency range
2. Sliding window: segments continuous signal into fixed-size windows
3. Repetition-based split: ensures no data leakage between train and test
"""

import numpy as np
from scipy.signal import butter, filtfilt


def apply_bandpass_filter(signal_data, lowcut=20.0, highcut=499.0, fs=1000.0, order=4):
    """
    Apply a Butterworth bandpass filter to remove noise from EMG signal.
    
    EMG useful frequency range: ~20-500 Hz.
    Below 20 Hz: movement artifacts, baseline wander
    Above 500 Hz: noise, interference
    
    Parameters:
    -----------
    signal_data : np.ndarray, shape (n_samples, n_channels)
        Raw EMG signal as a 2D array where each column is a channel.
    lowcut : float
        Low cutoff frequency in Hz. Frequencies BELOW this are removed.
    highcut : float
        High cutoff frequency in Hz. Frequencies ABOVE this are removed.
        MUST be less than fs/2 (the Nyquist frequency).
    fs : float
        Sampling frequency in Hz (samples per second).
    order : int
        Filter order. Higher = sharper cutoff but more potential signal distortion.
        Order 4 is standard for EMG.
    
    Returns:
    --------
    filtered_signal : np.ndarray, same shape as input
        Bandpass filtered signal with zero phase distortion.
    
    Technical Note:
    ---------------
    We use filtfilt() instead of lfilter() because filtfilt applies the filter
    forward AND backward, which cancels out the phase shift. This is important
    because phase shift would change the timing of muscle activations.
    """
    nyquist = 0.5 * fs  # Highest frequency we can represent (Nyquist theorem)
    
    # Convert Hz to normalized frequency (0 to 1, where 1 = Nyquist)
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design the Butterworth filter
    # b = numerator coefficients, a = denominator coefficients
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter forward and backward to eliminate phase shift
    # axis=0 means filter along the time axis (rows)
    filtered_signal = filtfilt(b, a, signal_data, axis=0)
    
    return filtered_signal


def sliding_window_with_labels(signal_data, labels, repetitions,
                                window_ms=200, overlap_pct=0.5, fs=200.0):
    """
    Segment continuous EMG into overlapping fixed-size windows.
    
    This is necessary because:
    1. EMG is a continuous stream, but classifiers need fixed-size inputs
    2. Overlapping windows provide more training data
    3. Each window captures enough temporal context (200ms) for gesture recognition
    
    Parameters:
    -----------
    signal_data : np.ndarray, shape (n_samples, n_channels)
        Filtered EMG signal. Each row is a time point, each column is a channel.
    labels : np.ndarray, shape (n_samples,)
        Gesture labels for each time point. Use restimulus, NOT stimulus.
        restimulus correctly marks rest periods between gestures as label 0.
    repetitions : np.ndarray, shape (n_samples,)
        Repetition number for each time point. This is CRITICAL for proper
        train/test splitting to avoid data leakage.
    window_ms : float
        Window duration in milliseconds.
        200ms is standard: long enough to capture muscle activation patterns,
        short enough for real-time control (<300ms delay required for prosthetics).
    overlap_pct : float
        Fraction of overlap between consecutive windows (0 to 1).
        0.5 (50%) is standard: provides more data and smoother transitions.
    fs : float
        Sampling frequency in Hz.
    
    Returns:
    --------
    X_windows : np.ndarray, shape (n_windows, window_samples, n_channels)
        3D array of windowed EMG segments.
    y_windows : np.ndarray, shape (n_windows,)
        Majority gesture label for each window.
    rep_windows : np.ndarray, shape (n_windows,)
        Majority repetition label for each window.
    
    Example:
    --------
    If fs=200Hz and window_ms=200ms:
    - window_samples = 40 samples
    - With 50% overlap: step = 20 samples
    - A 10-second recording produces ~1000 windows
    """
    # Convert milliseconds to number of samples
    window_samples = int((window_ms / 1000.0) * fs)
    step_samples = int(window_samples * (1.0 - overlap_pct))
    
    X_windows = []
    y_windows = []
    rep_windows = []
    
    total_samples = signal_data.shape[0]
    
    # Slide the window across the entire signal
    for start in range(0, total_samples - window_samples + 1, step_samples):
        end = start + window_samples
        
        # Extract the 2D window: (window_samples, n_channels)
        # This preserves the temporal structure within the window
        window = signal_data[start:end, :]
        X_windows.append(window)
        
        # Majority vote for gesture label
        # This handles the transition periods between gestures
        window_labels = labels[start:end]
        values, counts = np.unique(window_labels, return_counts=True)
        majority_label = values[np.argmax(counts)]
        y_windows.append(majority_label)
        
        # Majority vote for repetition label
        window_reps = repetitions[start:end]
        values_r, counts_r = np.unique(window_reps, return_counts=True)
        majority_rep = values_r[np.argmax(counts_r)]
        rep_windows.append(majority_rep)
    
    return (
        np.array(X_windows, dtype=np.float32),
        np.array(y_windows, dtype=np.int32),
        np.array(rep_windows, dtype=np.int32)
    )


def split_by_repetition(X, y, reps, test_reps=[2, 5]):
    """
    Split data by repetition for proper cross-validation.
    
    THIS IS THE MOST IMPORTANT FUNCTION FOR VALID EXPERIMENTAL RESULTS.
    
    Why repetition-based splitting?
    ------------------------------
    In NinaPro, each repetition is a separate recording session where the subject
    removes and re-dons the electrodes. If you randomly shuffle windows, windows
    from the SAME repetition can end up in both train and test sets. Since the
    electrode positions haven't changed within a repetition, the signal characteristics
    are nearly identical, giving you artificially high accuracy.
    
    Repetition-based splitting simulates real-world use where the user takes off
    and puts on the device again, causing electrode shift. This is the standard
    protocol in all EMG research.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_windows, window_samples, n_channels)
        Windowed EMG data.
    y : np.ndarray, shape (n_windows,)
        Gesture labels for each window.
    reps : np.ndarray, shape (n_windows,)
        Repetition labels for each window.
    test_reps : list of int
        Which repetition numbers to use for testing.
        For DB5 (6 reps): [2, 5] means train on reps 1,3,4,6 and test on reps 2,5
    
    Returns:
    --------
    X_train : np.ndarray
        Training windows (NOT from test_reps, and NOT rest class)
    X_test : np.ndarray
        Testing windows (from test_reps, and NOT rest class)
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Testing labels
    """
    # Create boolean masks
    test_mask = np.isin(reps, test_reps)
    train_mask = ~test_mask
    
    # Also filter out rest class (label 0) for classification
    # Rest is trivially easy to classify and inflates accuracy
    non_rest_mask = y > 0
    
    X_train = X[train_mask & non_rest_mask]
    y_train = y[train_mask & non_rest_mask]
    X_test = X[test_mask & non_rest_mask]
    y_test = y[test_mask & non_rest_mask]
    
    print(f"Train: {X_train.shape[0]} windows from reps {sorted(np.unique(reps[train_mask & non_rest_mask]))}")
    print(f"Test:  {X_test.shape[0]} windows from reps {sorted(np.unique(reps[test_mask & non_rest_mask]))}")
    print(f"Train classes: {sorted(np.unique(y_train))}")
    print(f"Test classes:  {sorted(np.unique(y_test))}")
    
    return X_train, X_test, y_train, y_test