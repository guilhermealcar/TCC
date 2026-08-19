import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy import stats
import h5py
import os

try:
    from src.config import DATABASE_CONFIGS, LOWCUT, HIGHCUT, NOTCH_FREQ, NOTCH_Q, REST_CLASS, REST_BALANCE_RATIO, PREPROCESSED_DIR
except ImportError:
    from config import DATABASE_CONFIGS, LOWCUT, HIGHCUT, NOTCH_FREQ, NOTCH_Q, REST_CLASS, REST_BALANCE_RATIO, PREPROCESSED_DIR

class sEMGPreprocessor:
    def __init__(self, database_name: str = "DB1"):
        self.config = DATABASE_CONFIGS[database_name]
        self.fs = self.config['fs']
        self.nyq = 0.5 * self.fs
        
        self.window_samples = int((self.config['window_ms'] / 1000.0) * self.fs)
        overlap_samples = int(self.window_samples * self.config['overlap_pct'])
        self.step_samples = self.window_samples - overlap_samples
        
        # Safe filter design: Adjust limits to not exceed Nyquist
        safe_low = min(LOWCUT, self.nyq - 5.0)
        safe_high = min(HIGHCUT, self.nyq - 1.0) 
        
        self.apply_bandpass = (safe_high > safe_low)
        if self.apply_bandpass:
            self.b_band, self.a_band = butter(2, [safe_low / self.nyq, safe_high / self.nyq], btype='band')
            
        # Only apply notch if Notch frequency is safely below Nyquist (prevents DB1 100Hz crash)
        self.apply_notch = (NOTCH_FREQ < self.nyq - 1.0)
        if self.apply_notch:
            self.b_notch, self.a_notch = iirnotch(NOTCH_FREQ, NOTCH_Q, self.fs)

    def filter_signal(self, emg_data: np.ndarray) -> np.ndarray:
        filtered = emg_data.copy()
        if self.apply_notch:
            filtered = filtfilt(self.b_notch, self.a_notch, filtered, axis=0)
        if self.apply_bandpass:
            filtered = filtfilt(self.b_band, self.a_band, filtered, axis=0)
        return filtered

    def standardize(self, emg_data: np.ndarray) -> np.ndarray:
        """Channel-wise Z-score standardization."""
        mean = np.mean(emg_data, axis=0)
        std = np.std(emg_data, axis=0)
        return (emg_data - mean) / (std + 1e-8)

    def extract_windows(self, emg: np.ndarray, labels: np.ndarray, reps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Slices continuous arrays into sliding windows with majority voting for labels."""
        n_samples, n_channels = emg.shape
        n_windows = (n_samples - self.window_samples) // self.step_samples + 1
        
        X_windows = np.zeros((n_windows, self.window_samples, n_channels))
        y_windows = np.zeros(n_windows)
        rep_windows = np.zeros(n_windows)
        
        for i in range(n_windows):
            start = i * self.step_samples
            end = start + self.window_samples
            X_windows[i] = emg[start:end, :]
            
            y_windows[i] = stats.mode(labels[start:end], keepdims=False)[0]
            rep_windows[i] = stats.mode(reps[start:end], keepdims=False)[0]
            
        return X_windows, y_windows, rep_windows

    def balance_rest_class(self, X: np.ndarray, y: np.ndarray, reps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Undersamples Class 0 based on the average size of active gesture classes."""
        rest_idx = np.where(y == REST_CLASS)[0]
        active_idx = np.where(y != REST_CLASS)[0]
        
        if len(active_idx) == 0:
            return X, y, reps
            
        _, counts = np.unique(y[active_idx], return_counts=True)
        avg_active_size = int(np.mean(counts))
        target_rest_size = int(avg_active_size * REST_BALANCE_RATIO)
        
        if len(rest_idx) > target_rest_size:
            np.random.seed(42)
            sampled_rest_idx = np.random.choice(rest_idx, target_rest_size, replace=False)
            balanced_idx = np.sort(np.concatenate([sampled_rest_idx, active_idx]))
            return X[balanced_idx], y[balanced_idx], reps[balanced_idx]
            
        return X, y, reps

    def extract_dense_windows(self, emg_data: np.ndarray, labels: np.ndarray, reps: np.ndarray, window_size: int = 20) -> tuple:
        """
        [main_V4] Extracts highly dense overlapping windows using a 10ms stride (1 sample at 100Hz).
        This exponentially scales the dataset size and provides shift-invariant training data 
        for state-of-the-art Deep Learning and Dictionary Learning architectures.
        
        Args:
            emg_data: Standardized continuous sEMG signal (n_samples, n_channels)
            labels: Continuous label array (n_samples,)
            reps: Continuous repetition array (n_samples,)
            window_size: Number of samples per window (default 20 = 200ms at 100Hz)
            
        Returns:
            X_win: 3D tensor of dense windows (n_windows, window_size, n_channels)
            y_win: 1D array of labels for each window
            reps_win: 1D array of repetitions for each window
        """
        print(f"Extracting DENSE windows (stride=1 sample / 10ms)...")
        from numpy.lib.stride_tricks import sliding_window_view
        
        # sliding_window_view creates a memory-efficient view of the data.
        # Initial shape: (num_windows, n_channels, window_size)
        X_view = sliding_window_view(emg_data, window_shape=window_size, axis=0)
        
        # Transpose to match our PyTorch/DL standard: (num_windows, window_size, n_channels)
        X_win = X_view.transpose(0, 2, 1).copy() # .copy() forces allocation so it's safe to modify/save
        
        # For a causal system (real-time prediction), the label of a window 
        # is determined by the physical gesture occurring at the *end* of that window.
        y_win = labels[window_size - 1:]
        reps_win = reps[window_size - 1:]
        
        print(f"Dense extraction complete. Exploded dataset to {X_win.shape[0]} windows (Shape: {X_win.shape}).")
        return X_win, y_win, reps_win

def save_preprocessed_hdf5(subject_id: int, X: np.ndarray, y: np.ndarray, reps: np.ndarray, db_name: str = "DB1") -> None:
    """Serializes arrays into a compressed HDF5 dataset."""
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    file_path = os.path.join(PREPROCESSED_DIR, f"{db_name}_subject_{subject_id}.h5")
    
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('X', data=X, compression='gzip', chunks=True)
        f.create_dataset('y', data=y, compression='gzip', chunks=True)
        f.create_dataset('reps', data=reps, compression='gzip', chunks=True)