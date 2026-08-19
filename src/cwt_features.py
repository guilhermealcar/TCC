import numpy as np
import pywt
from joblib import Parallel, delayed

def process_cwt_window(window: np.ndarray, scales: np.ndarray, wavelet_name: str) -> np.ndarray:
    """
    Applies CWT to a single window across all channels.
    window shape: (n_samples, n_channels)
    Returns shape: (n_scales, n_samples, n_channels)
    """
    n_samples, n_channels = window.shape
    n_scales = len(scales)
    
    # Initialize the spectrogram matrix
    cwt_window = np.zeros((n_scales, n_samples, n_channels))
    
    for c in range(n_channels):
        signal = window[:, c]
        # Extract the complex wavelet coefficients
        coefs, _ = pywt.cwt(signal, scales, wavelet_name)
        # We only care about the energy magnitude (absolute value)
        cwt_window[:, :, c] = np.abs(coefs)
        
    return cwt_window

def extract_cwt_spectrograms(X_windows: np.ndarray, n_scales: int = 16, n_jobs: int = -1) -> np.ndarray:
    """
    Extracts 2D Time-Frequency spectrograms using the Complex Morlet Wavelet.
    Input shape: (n_windows, window_samples, n_channels)
    Output shape: (n_windows, n_scales, window_samples, n_channels)
    """
    print(f"Starting parallel CWT extraction on {X_windows.shape[0]} windows...")
    
    # Define the scales (frequencies) to extract. 
    # 1 to 16 covers the primary energy bands of sEMG signals.
    scales = np.arange(1, n_scales + 1)
    wavelet = 'cmor1.5-1.0' 
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_cwt_window)(X_windows[i], scales, wavelet) 
        for i in range(X_windows.shape[0])
    )
    
    cwt_tensor = np.stack(results, axis=0)
    print(f"CWT extraction complete. Spectrogram tensor shape: {cwt_tensor.shape}")
    
    return cwt_tensor