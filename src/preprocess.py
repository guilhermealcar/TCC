# src/preprocess.py
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

def load_ninapro_mat(file_path: str) -> dict:
    """Loads raw NinaPro .mat file."""
    mat_data = sio.loadmat(file_path)
    return {
        'emg': mat_data['emg'],
        'labels': mat_data['restimulus'].squeeze(),
        'reps': mat_data['rerepetition'].squeeze()
    }

class sEMGPreprocessor:
    def __init__(self, sample_rate: int = 2000):
        self.fs = sample_rate
        self.mean_ = None
        self.std_ = None

    def filter_signal(self, data: np.ndarray, btype='bandpass', lowcut=20.0, highcut=500.0, order=4) -> np.ndarray:
        """
        Applies a Butterworth filter. 
        Molina et al. (2025) standard: 20-500Hz band-pass for 2000Hz DB2 data.
        """
        nyq = 0.5 * self.fs
        
        if btype == 'bandpass':
            # Edge case handling for DB1 (100Hz)
            if highcut >= nyq:
                highcut = nyq - 1.0 
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='bandpass')
        elif btype == 'high':
            low = lowcut / nyq
            b, a = butter(order, low, btype='high')
            
        return filtfilt(b, a, data, axis=0)

    def fit_standardize(self, data: np.ndarray) -> np.ndarray:
        """Fits standardization parameters and applies them."""
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        # Prevent division by zero on flatlines
        self.std_[self.std_ == 0] = 1.0 
        return (data - self.mean_) / self.std_

    def standardize(self, data: np.ndarray) -> np.ndarray:
        """Applies previously fitted standardization."""
        return (data - self.mean_) / self.std_

    def extract_dense_windows(self, emg: np.ndarray, labels: np.ndarray, reps: np.ndarray, win_size: int, stride: int) -> tuple:
        """
        Extracts sliding windows based on configurable size and stride.
        Molina standard: 500 samples (250ms), 100 stride (50ms).
        Outputs: (N_windows, win_size, channels).
        """
        print(f"Extracting windows (size={win_size} samples, stride={stride} samples)...")
        
        num_windows = (emg.shape[0] - win_size) // stride + 1
        indices = np.arange(num_windows) * stride
        
        X_win = np.array([emg[i : i + win_size] for i in indices])
        
        # The label is determined by the gesture at the very end of the time window
        y_win = np.array([labels[i + win_size - 1] for i in indices])
        reps_win = np.array([reps[i + win_size - 1] for i in indices])
        
        return X_win, y_win, reps_win

    def balance_rest_class(self, X: np.ndarray, y: np.ndarray, reps: np.ndarray) -> tuple:
        """Subsamples the massive Rest class (0) to match the average active class size."""
        active_idx = np.where(y > 0)[0]
        rest_idx = np.where(y == 0)[0]
        
        if len(active_idx) == 0:
            return X, y, reps
            
        unique_classes = np.unique(y[active_idx])
        avg_active_size = len(active_idx) // len(unique_classes)
        
        target_rest_size = min(len(rest_idx), avg_active_size)
        
        np.random.seed(42)
        subsampled_rest_idx = np.random.choice(rest_idx, size=target_rest_size, replace=False)
        
        balanced_idx = np.concatenate([active_idx, subsampled_rest_idx])
        balanced_idx = np.sort(balanced_idx)
        
        return X[balanced_idx], y[balanced_idx], reps[balanced_idx]