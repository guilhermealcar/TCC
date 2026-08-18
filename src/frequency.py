import numpy as np
from scipy.fft import fft

def extract_fft_magnitude(X_windows: np.ndarray) -> np.ndarray:
    """
    Transforms time-domain windows into frequency-domain magnitude spectra.
    Input shape: (n_windows, window_samples, n_channels)
    
    Returns:
        fft_magnitude: (n_windows, window_samples // 2, n_channels)
    """
    print(f"Extracting FFT from input shape {X_windows.shape}...")
    n_windows, n_samples, n_channels = X_windows.shape
    
    # Compute the 1D Discrete Fourier Transform along the time axis (axis=1)
    fft_complex = fft(X_windows, axis=1)
    
    # We only care about the magnitude (amplitude of frequencies)
    fft_magnitude = np.abs(fft_complex)
    
    # By Nyquist's theorem, the second half of the FFT is perfectly symmetrical 
    # for real signals, so we slice it in half to remove redundant data.
    half_point = n_samples // 2
    fft_positive = fft_magnitude[:, :half_point, :]
    
    print(f"FFT extraction complete. Output shape: {fft_positive.shape}")
    return fft_positive