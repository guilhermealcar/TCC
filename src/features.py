"""
Classical time-domain features for EMG signal analysis.

These features serve as a BASELINE for comparison against dictionary learning.
For each EMG window, we extract 5 features per channel, giving n_channels * 5 total features.

Features extracted:
1. MAV (Mean Absolute Value): Average muscle activation level
2. RMS (Root Mean Square): Related to muscle force/power
3. WL (Waveform Length): Combined measure of amplitude and frequency
4. ZC (Zero Crossings): Simple frequency indicator
5. SSC (Slope Sign Changes): Another frequency indicator

For RECONSTRUCTION comparison: since classical features lose temporal information,
we attempt reconstruction using the pseudo-inverse of the feature extraction process.
This is an approximation and will generally perform worse than dictionary methods.
"""

import numpy as np


def extract_time_domain_features(window_data):
    """
    Extract 5 classical time-domain features from a single EMG window.
    
    Parameters:
    -----------
    window_data : np.ndarray, shape (window_samples, n_channels)
        A single EMG window (2D array: time × channels).
    
    Returns:
    --------
    features : np.ndarray, shape (n_channels * 5,)
        Concatenated features: [MAV_ch1, ..., MAV_chN, RMS_ch1, ..., RMS_chN, ...]
    
    Feature Details:
    ----------------
    MAV = (1/N) * Σ|x_i|
        - Measures average activation level
        - Simple, computationally efficient
    
    RMS = sqrt((1/N) * Σx_i²)
        - Measures signal power
        - Directly related to muscle force
    
    WL = Σ|x_i - x_{i-1}|
        - Measures cumulative signal variation
        - Captures both amplitude and frequency information
    
    ZC = count of sign changes
        - Simple frequency measure
        - Counts how often signal crosses zero
    
    SSC = count of slope sign changes
        - More robust frequency measure than ZC
        - Counts changes in the derivative's sign
    """
    # Mean Absolute Value
    mav = np.mean(np.abs(window_data), axis=0)
    
    # Root Mean Square
    rms = np.sqrt(np.mean(window_data ** 2, axis=0))
    
    # Waveform Length (cumulative absolute difference)
    wl = np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)
    
    # Zero Crossings (count sign changes)
    zc = np.sum(np.diff(np.sign(window_data), axis=0) != 0, axis=0)
    
    # Slope Sign Changes (count sign changes in the derivative)
    diff_signal = np.diff(window_data, axis=0)
    ssc = np.sum(np.diff(np.sign(diff_signal), axis=0) != 0, axis=0)
    
    return np.hstack([mav, rms, wl, zc, ssc])


def extract_features_batch(X_windows):
    """
    Extract classical features for all windows.
    
    Parameters:
    -----------
    X_windows : np.ndarray, shape (n_windows, window_samples, n_channels)
        3D array of EMG windows.
    
    Returns:
    --------
    features : np.ndarray, shape (n_windows, n_channels * 5)
        2D feature matrix.
    """
    n_windows = X_windows.shape[0]
    features_list = []
    
    for i in range(n_windows):
        features_list.append(extract_time_domain_features(X_windows[i]))
    
    return np.array(features_list)


# ============================================================================
# Classical Features Reconstruction (for comparison)
# ============================================================================

def build_feature_extraction_matrix(n_channels, window_samples):
    """
    Build the linear transformation matrix that maps a flattened window
    to its classical features.
    
    This matrix F satisfies: features = F @ window_flat
    
    The extraction is linear except for ZC and SSC (which are non-linear).
    For reconstruction, we use a pseudo-inverse of this matrix.
    
    Parameters:
    -----------
    n_channels : int
        Number of EMG channels.
    window_samples : int
        Number of time samples per window.
    
    Returns:
    --------
    F : np.ndarray, shape (n_features, n_channels * window_samples)
        Feature extraction matrix (approximate, linear part only).
    
    Note:
    -----
    This is an APPROXIMATION because ZC and SSC are non-linear operations.
    The reconstruction from classical features will therefore be imperfect,
    which is expected and demonstrates the advantage of dictionary learning.
    """
    signal_dim = n_channels * window_samples
    n_features = n_channels * 5  # MAV, RMS, WL, ZC, SSC per channel
    
    F = np.zeros((n_features, signal_dim))
    
    for ch in range(n_channels):
        # Indices for this channel in the flattened window
        start_idx = ch * window_samples
        end_idx = (ch + 1) * window_samples
        
        # MAV row: (1/N) * |x| (using absolute value approximation)
        F[ch + 0 * n_channels, start_idx:end_idx] = 1.0 / window_samples
        
        # RMS row: we approximate as linear for reconstruction
        # True RMS is non-linear, but we use the linear part
        F[ch + 1 * n_channels, start_idx:end_idx] = np.sqrt(1.0 / window_samples)
        
        # WL row: difference operator
        for t in range(window_samples - 1):
            F[ch + 2 * n_channels, start_idx + t] = -1.0
            F[ch + 2 * n_channels, start_idx + t + 1] = 1.0
        
        # ZC and SSC are highly non-linear, so we set them to approximate linear forms
        # This is a limitation of the classical features for reconstruction
        F[ch + 3 * n_channels, start_idx:end_idx] = 1.0 / window_samples
        F[ch + 4 * n_channels, start_idx:end_idx] = 1.0 / window_samples
    
    return F


def classical_features_to_signal(classical_features, n_channels, window_samples):
    """
    Attempt to reconstruct the original signal from classical features.
    
    This uses the Moore-Penrose pseudo-inverse of the feature extraction matrix.
    The reconstruction is APPROXIMATE because:
    1. ZC and SSC are non-linear and lose information
    2. The mapping from signal to features is many-to-one
    
    Parameters:
    -----------
    classical_features : np.ndarray, shape (n_windows, n_features)
        Classical features extracted from EMG windows.
    n_channels : int
        Number of EMG channels.
    window_samples : int
        Number of time samples per window.
    
    Returns:
    --------
    reconstructed : np.ndarray, shape (n_windows, window_samples, n_channels)
        Approximate reconstruction of the original signals.
    """
    n_windows = classical_features.shape[0]
    signal_dim = n_channels * window_samples
    
    # Build the feature extraction matrix
    F = build_feature_extraction_matrix(n_channels, window_samples)
    
    # Compute pseudo-inverse
    # F_pinv = (F^T F)^{-1} F^T
    # This gives the minimum-norm least-squares solution
    F_pinv = np.linalg.pinv(F)
    
    # Reconstruct: y ≈ F_pinv @ features
    reconstructed_flat = classical_features @ F_pinv.T
    
    # Reshape to 3D
    reconstructed = reconstructed_flat.reshape(n_windows, window_samples, n_channels)
    
    return reconstructed