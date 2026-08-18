import numpy as np
from vmdpy import VMD
from joblib import Parallel, delayed

def process_single_window(window: np.ndarray, K: int, alpha: int, tau: float, DC: int, init: int, tol: float) -> np.ndarray:
    """
    Applies VMD to a single spatial-temporal window.
    window shape: (n_samples, n_channels)
    Returns shape: (K, n_samples, n_channels)
    """
    n_samples, n_channels = window.shape
    imfs_window = np.zeros((K, n_samples, n_channels))
    
    # Process each electrode channel independently
    for c in range(n_channels):
        signal = window[:, c]
        # VMD returns: u (IMFs), u_hat (spectra), omega (center frequencies)
        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
        imfs_window[:, :, c] = u
        
    return imfs_window

def extract_vmd_features(X_windows: np.ndarray, K: int = 3, n_jobs: int = -1) -> np.ndarray:
    """
    Extracts VMD Intrinsic Mode Functions (IMFs) for the entire dataset in parallel.
    Input shape: (n_windows, window_samples, n_channels)
    
    Returns:
        vmd_tensor: (n_windows, K, window_samples, n_channels)
    """
    print(f"Starting parallel VMD extraction on {X_windows.shape[0]} windows...")
    print(f"Extracting K={K} modes per channel. This may take a few minutes...")
    
    # VMD Optimization Parameters (Standard defaults for physiological signals)
    alpha = 2000       # moderate bandwidth constraint
    tau = 0.           # noise-tolerance (no strict fidelity enforcement)
    DC = 0             # no DC part imposed
    init = 1           # initialize omegas uniformly
    tol = 1e-7         # tolerance of convergence
    
    # Parallelize across windows using all available CPU cores
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_window)(X_windows[i], K, alpha, tau, DC, init, tol) 
        for i in range(X_windows.shape[0])
    )
    
    vmd_tensor = np.stack(results, axis=0)
    print(f"VMD extraction complete. Output tensor shape: {vmd_tensor.shape}")
    
    return vmd_tensor