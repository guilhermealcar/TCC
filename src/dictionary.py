import os
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder

try:
    from src.config import MODELS_DIR
except ImportError:
    from config import MODELS_DIR

class ChannelWiseDictionaryLearner:
    def __init__(self, n_atoms: int = 64, n_nonzero_coefs: int = 5, batch_size: int = 1024):
        self.n_atoms = n_atoms
        self.n_nonzero_coefs = n_nonzero_coefs
        self.batch_size = batch_size
        
        self.dict_learner = MiniBatchDictionaryLearning(
            n_components=self.n_atoms,
            alpha=1.0,
            batch_size=self.batch_size,
            fit_algorithm='lars',
            transform_algorithm='omp',
            transform_n_nonzero_coefs=self.n_nonzero_coefs,
            positive_dict=True, 
            random_state=42
        )
        self.dictionary_ = None

    def fit(self, X_windows: np.ndarray) -> None:
        """
        Expects tensor of shape (n_windows, window_samples, n_channels).
        Learns 1D atoms by treating every channel as an independent temporal observation.
        """
        n_windows, n_samples, n_channels = X_windows.shape
        
        # Transpose and reshape to (n_windows * n_channels, window_samples)
        # We process each channel's 20-sample window individually
        X_1d = X_windows.transpose(0, 2, 1).reshape(-1, n_samples)
        
        print(f"Training 1D Dictionary with {self.n_atoms} atoms on {X_1d.shape[0]} temporal windows...")
        self.dict_learner.fit(X_1d)
        self.dictionary_ = self.dict_learner.components_
        print("Channel-wise dictionary learning complete.")

    def save_dictionary(self, filename: str = "emg_dict_1D_S1.npz") -> None:
        if self.dictionary_ is None:
            raise ValueError("Dictionary has not been trained yet.")
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, filename)
        np.savez(path, dictionary=self.dictionary_)
        print(f"1D Dictionary saved to {path}")

    def load_dictionary(self, filename: str = "emg_dict_1D_S1.npz") -> None:
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dictionary file not found: {path}")
        data = np.load(path)
        self.dictionary_ = data['dictionary']
        self.dict_learner.components_ = self.dictionary_
        print(f"1D Dictionary loaded from {path}")


class ChannelWiseOMPExtractor:
    def __init__(self, dictionary: np.ndarray, n_nonzero_coefs: int = 5):
        self.dictionary = dictionary
        self.n_nonzero_coefs = n_nonzero_coefs
        self.coder = SparseCoder(
            dictionary=self.dictionary,
            transform_algorithm='omp',
            transform_n_nonzero_coefs=self.n_nonzero_coefs
        )

    def transform(self, X_windows: np.ndarray) -> np.ndarray:
        """
        Projects each channel independently and concatenates the sparse codes.
        Returns shape: (n_windows, n_channels * n_atoms)
        """
        n_windows, n_samples, n_channels = X_windows.shape
        n_atoms = self.dictionary.shape[0]
        
        # Flatten to 1D windows
        X_1d = X_windows.transpose(0, 2, 1).reshape(-1, n_samples)
        
        # Extract sparse codes for every single channel independently
        sparse_1d = self.coder.transform(X_1d)
        
        # Reshape back to (n_windows, n_channels, n_atoms)
        sparse_3d = sparse_1d.reshape(n_windows, n_channels, n_atoms)
        
        # Flatten channels and atoms into the final feature vector per window
        return sparse_3d.reshape(n_windows, -1)