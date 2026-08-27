# src/features.py

import numpy as np
import joblib
from sklearn.decomposition import MiniBatchDictionaryLearning
import warnings

# Suppress OMP linear dependence warnings which are normal in sparse coding
warnings.filterwarnings("ignore", message="Orthogonal matching pursuit ended prematurely")

def extract_fft_magnitude(X_windows: np.ndarray) -> np.ndarray:
    """
    Applies the Real Fast Fourier Transform (rFFT) along the time axis.
    Input: (N_windows, 20 time_steps, 10 channels)
    Output: (N_windows, 11 freq_bins, 10 channels)
    """
    print("Extracting FFT magnitudes...")
    # rfft returns (window_size // 2) + 1 bins. For 20 samples, this is 11 bins.
    fft_vals = np.fft.rfft(X_windows, axis=1)
    return np.abs(fft_vals)

class EMGDictionaryLearner:
    """
    Learns sparse physical atoms from frequency spectra and transforms 
    them into highly compressed Sparse Codes for classification.
    """
    def __init__(self, n_atoms: int = 128, n_nonzero_coefs: int = 5, random_state: int = 42):
        self.n_atoms = n_atoms
        self.n_nonzero_coefs = n_nonzero_coefs
        self.dict_learner = MiniBatchDictionaryLearning(
            n_components=n_atoms,
            transform_algorithm='omp',
            transform_n_nonzero_coefs=n_nonzero_coefs,
            random_state=random_state,
            batch_size=1024,
            max_iter=100
        )

    def fit(self, X_freq: np.ndarray):
        """Fits the dictionary on the flattened frequency-channel data."""
        N, F, C = X_freq.shape
        X_flat = X_freq.reshape(N, F * C)
        print(f"Fitting Dictionary ({self.n_atoms} atoms) on shape {X_flat.shape}...")
        self.dict_learner.fit(X_flat)
        print("Dictionary learning complete.")

    def transform(self, X_freq: np.ndarray) -> np.ndarray:
        """Transforms frequency data into a sparse feature matrix."""
        N, F, C = X_freq.shape
        X_flat = X_freq.reshape(N, F * C)
        return self.dict_learner.transform(X_flat)

    def save(self, filepath: str):
        joblib.dump(self.dict_learner, filepath)

    def load(self, filepath: str):
        self.dict_learner = joblib.load(filepath)

def extract_time_domain_features(X_windows: np.ndarray) -> np.ndarray:
    """
    Extracts essential Time-Domain features (MAV and RMS) for each channel.
    Input: (N_windows, 20 time_steps, 10 channels)
    Output: (N_windows, 20 features) -> 2 features per channel
    """
    print("Extracting Time-Domain features (MAV, RMS)...")
    # Mean Absolute Value
    mav = np.mean(np.abs(X_windows), axis=1)
    
    # Root Mean Square
    rms = np.sqrt(np.mean(X_windows**2, axis=1))
    
    # Concatenate features horizontally
    return np.hstack((mav, rms))