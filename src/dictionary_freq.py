import os
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
from ksvd import ApproximateKSVD

try:
    from .config import MODELS_DIR
except ImportError:
    from config import MODELS_DIR

class FrequencyDictionaryLearner:
    def __init__(self, n_atoms: int = 32, transform_n_nonzero_coefs: int = 3, method: str = 'ksvd'):
        """
        method: 'ksvd' or 'minibatch'
        """
        self.n_atoms = n_atoms
        self.n_nonzero_coefs = transform_n_nonzero_coefs
        self.method = method
        self.dictionary_ = None
        
        if self.method == 'minibatch':
            self.model = MiniBatchDictionaryLearning(
                n_components=self.n_atoms,
                alpha=1.0,
                batch_size=1024,
                fit_algorithm='lars',
                transform_algorithm='omp',
                transform_n_nonzero_coefs=self.n_nonzero_coefs,
                positive_dict=True,
                random_state=42
            )
        elif self.method == 'ksvd':
            self.model = ApproximateKSVD(
                n_components=self.n_atoms,
                transform_n_nonzero_coefs=self.n_nonzero_coefs
            )
        else:
            raise ValueError("Method must be strictly 'ksvd' or 'minibatch'")

    def fit(self, X_freq: np.ndarray) -> None:
        """
        Expects tensor of shape (n_windows, freq_bins, n_channels).
        Learns 1D frequency atoms channel-wise.
        """
        n_windows, n_bins, n_channels = X_freq.shape
        
        # Reshape to treat each channel's frequency spectrum as an independent observation
        X_1d = X_freq.transpose(0, 2, 1).reshape(-1, n_bins)
        
        print(f"Training {self.method.upper()} Dictionary with {self.n_atoms} atoms on {X_1d.shape[0]} frequency spectra...")
        self.model.fit(X_1d)
        self.dictionary_ = self.model.components_
        print("Dictionary learning complete.")

    def save_dictionary(self, filename: str) -> None:
        if self.dictionary_ is None:
            raise ValueError("Dictionary has not been trained yet.")
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, filename)
        np.savez(path, dictionary=self.dictionary_)
        print(f"Dictionary saved to {path}")

    def load_dictionary(self, filename: str) -> None:
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dictionary file not found: {path}")
        data = np.load(path)
        self.dictionary_ = data['dictionary']
        
        if self.method == 'minibatch':
            self.model.components_ = self.dictionary_
        elif self.method == 'ksvd':
            self.model.dictionary = self.dictionary_
            
        print(f"Dictionary loaded from {path}")


class FrequencyOMPExtractor:
    def __init__(self, dictionary: np.ndarray, n_nonzero_coefs: int = 3):
        self.dictionary = dictionary
        self.n_nonzero_coefs = n_nonzero_coefs
        self.coder = SparseCoder(
            dictionary=self.dictionary,
            transform_algorithm='omp',
            transform_n_nonzero_coefs=self.n_nonzero_coefs
        )

    def transform(self, X_freq: np.ndarray) -> np.ndarray:
        """
        Projects each channel's frequency spectrum independently and concatenates the sparse codes.
        Returns shape: (n_windows, n_channels * n_atoms)
        """
        n_windows, n_bins, n_channels = X_freq.shape
        n_atoms = self.dictionary.shape[0]
        
        # Flatten to 1D spectra
        X_1d = X_freq.transpose(0, 2, 1).reshape(-1, n_bins)
        
        # Extract sparse codes
        sparse_1d = self.coder.transform(X_1d)
        
        # Reshape back to maintain spatial channel separation
        sparse_3d = sparse_1d.reshape(n_windows, n_channels, n_atoms)
        
        return sparse_3d.reshape(n_windows, -1)