import numpy as np

def extract_hudgins_features(X_windows: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    """
    Extracts vectorized Hudgins features: MAV, RMS, WL, ZC, SSC.
    X_windows shape: (n_windows, window_samples, n_channels)
    Returns shape: (n_windows, n_channels * 5)
    """
    print(f"Extracting Hudgins features from shape {X_windows.shape}...")
    
    # 1. Mean Absolute Value (MAV)
    mav = np.mean(np.abs(X_windows), axis=1)
    
    # 2. Root Mean Square (RMS)
    rms = np.sqrt(np.mean(X_windows**2, axis=1))
    
    # 3. Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(X_windows, axis=1)), axis=1)
    
    # 4. Zero Crossings (ZC)
    diff_sign = np.diff(np.sign(X_windows), axis=1)
    abs_diff = np.abs(np.diff(X_windows, axis=1))
    zc = np.sum((np.abs(diff_sign) > 0) & (abs_diff >= threshold), axis=1)
    
    # 5. Slope Sign Changes (SSC)
    diff_signal = np.diff(X_windows, axis=1)
    diff_diff_sign = np.diff(np.sign(diff_signal), axis=1)
    abs_diff_align = np.abs(diff_signal[:, 1:, :]) 
    abs_diff_prev_align = np.abs(diff_signal[:, :-1, :])
    
    ssc = np.sum(
        (np.abs(diff_diff_sign) > 0) & 
        ((abs_diff_align >= threshold) | (abs_diff_prev_align >= threshold)), 
        axis=1
    )
    
    # Concatenate features horizontally: shape (n_windows, 5 * n_channels)
    features = np.hstack([mav, rms, wl, zc, ssc])
    print(f"Feature extraction complete. Output shape: {features.shape}")
    return features