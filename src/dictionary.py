"""
Dictionary Learning for EMG signal representation.

Two algorithms are implemented:
1. MiniBatchDictionaryLearning (Mairal et al., 2009): Fast, stochastic gradient-based
2. K-SVD (Aharon et al., 2006): Classic, SVD-based, more accurate but slower

Both learn an overcomplete dictionary D where:
- Each row is an "atom" (a basic signal pattern)
- Any EMG window y can be approximated as y ≈ D^T · x
- x is sparse (mostly zeros) — only k atoms are used
"""

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit

# Try to import K-SVD, but don't crash if not installed
try:
    from ksvd import ApproximateKSVD
    KSVD_AVAILABLE = True
except ImportError:
    print("WARNING: ksvd package not installed. K-SVD will not be available.")
    print("Install with: pip install ksvd")
    KSVD_AVAILABLE = False


# ============================================================================
# PART 1: Dictionary Training Functions
# ============================================================================

def train_minibatch_dictionary(X_train_windows, n_components=100, alpha=1.0,
                                batch_size=32, max_iter=1000):
    """
    Train a dictionary using MiniBatchDictionaryLearning (online algorithm).
    
    This algorithm processes data in small batches, making it fast and memory-efficient.
    It's suitable for large datasets. The optimization uses stochastic gradient descent
    to minimize:
    
        min ||X - D·A||² + alpha * ||A||₁
    
    where X is the data, D is the dictionary, and A is the sparse codes.
    
    Parameters:
    -----------
    X_train_windows : np.ndarray, shape (n_windows, window_samples, n_channels)
        3D array of EMG windows from training set.
    n_components : int
        Number of dictionary atoms. This is the compression dimension.
        - n_components < signal_dim: undercomplete (compression)
        - n_components > signal_dim: overcomplete (can capture more patterns)
        Typical: 50-200 for EMG
    alpha : float
        Sparsity regularization parameter. Higher = sparser codes.
        This controls the L1 penalty on the coefficients.
    batch_size : int
        Number of samples per mini-batch. Smaller = noisier updates but faster.
    max_iter : int
        Maximum number of iterations over the data.
    
    Returns:
    --------
    dict_learner : MiniBatchDictionaryLearning object
        Trained dictionary learner. Use dict_learner.transform() to get sparse codes.
    dictionary : np.ndarray, shape (n_components, signal_dim)
        The learned dictionary atoms. Each row is one atom (a basic signal pattern).
    """
    n_windows = X_train_windows.shape[0]
    
    # Flatten each window from 3D to 1D
    # (n_windows, window_samples, n_channels) -> (n_windows, window_samples * n_channels)
    X_flat = X_train_windows.reshape(n_windows, -1)
    signal_dim = X_flat.shape[1]
    
    print(f"Training MiniBatch Dictionary...")
    print(f"  Samples: {n_windows}")
    print(f"  Signal dimension: {signal_dim}")
    print(f"  Atoms: {n_components}")
    print(f"  Compression ratio: {signal_dim / n_components:.1f}:1")
    
    dict_learner = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=42,
        transform_algorithm='omp'  # Use OMP for sparse coding
    )
    dict_learner.fit(X_flat)
    
    return dict_learner, dict_learner.components_


def train_ksvd_dictionary(X_train_windows, n_components=100,
                           transform_n_nonzero_coefs=5, max_iter=10):
    """
    Train a dictionary using K-SVD (K-Singular Value Decomposition).
    
    K-SVD is the classic dictionary learning algorithm. It iterates between:
    1. Sparse Coding: Fix D, find sparse codes X using OMP
    2. Dictionary Update: Fix X, update each atom d_j using SVD on the residual
    
    This is more accurate than MiniBatch but slower for large datasets.
    
    Parameters:
    -----------
    X_train_windows : np.ndarray, shape (n_windows, window_samples, n_channels)
        3D array of EMG windows from training set.
    n_components : int
        Number of dictionary atoms (compression dimension).
    transform_n_nonzero_coefs : int
        Exact number of non-zero coefficients (sparsity level k).
        In K-SVD, we control sparsity directly (L0 norm) rather than with alpha (L1 norm).
    max_iter : int
        Number of K-SVD iterations. Each iteration does sparse coding + dictionary update.
        10-20 is typical. More iterations = better dictionary but slower.
    
    Returns:
    --------
    dict_learner : ApproximateKSVD object
        Trained K-SVD learner.
    dictionary : np.ndarray, shape (n_components, signal_dim)
        The learned dictionary atoms.
    """
    if not KSVD_AVAILABLE:
        raise ImportError("K-SVD requires the 'ksvd' package. Install with: pip install ksvd")
    
    n_windows = X_train_windows.shape[0]
    
    # Flatten each window
    X_flat = X_train_windows.reshape(n_windows, -1)
    signal_dim = X_flat.shape[1]
    
    print(f"Training K-SVD Dictionary...")
    print(f"  Samples: {n_windows}")
    print(f"  Signal dimension: {signal_dim}")
    print(f"  Atoms: {n_components}")
    print(f"  Sparsity (k): {transform_n_nonzero_coefs}")
    print(f"  Compression ratio: {signal_dim / n_components:.1f}:1")
    
    dict_learner = ApproximateKSVD(
        n_components=n_components,
        transform_n_nonzero_coefs=transform_n_nonzero_coefs,
        max_iter=max_iter
    )
    dict_learner.fit(X_flat)
    
    return dict_learner, dict_learner.components_


# ============================================================================
# PART 2: Sparse Coding (Encoding)
# ============================================================================

def encode_signals(X_windows, dict_learner, n_nonzero=5):
    """
    Encode EMG windows into sparse codes using Orthogonal Matching Pursuit (OMP).
    
    This is the COMPRESSION step. Each window of shape (window_samples * n_channels)
    is compressed into a sparse vector of length n_atoms, where only k entries are non-zero.
    
    The sparse code x is found by solving:
        min ||x||₀  subject to  ||y - D^T·x||² ≤ ε
    where ||x||₀ is the number of non-zero entries.
    
    Parameters:
    -----------
    X_windows : np.ndarray, shape (n_windows, window_samples, n_channels)
        EMG windows to encode.
    dict_learner : trained dictionary object
        Either MiniBatchDictionaryLearning or ApproximateKSVD.
    n_nonzero : int
        Maximum number of non-zero coefficients (sparsity level k).
        This is the compression: you represent the signal using only k numbers
        (plus which atoms they correspond to).
    
    Returns:
    --------
    sparse_codes : np.ndarray, shape (n_windows, n_atoms)
        Sparse representation of each window. Only k entries per row are non-zero.
    """
    n_windows = X_windows.shape[0]
    X_flat = X_windows.reshape(n_windows, -1)
    
    # Use the dictionary's transform method (which uses OMP internally)
    sparse_codes = dict_learner.transform(X_flat)
    
    # Verify sparsity
    avg_nonzero = np.mean(np.sum(sparse_codes != 0, axis=1))
    print(f"  Encoded {n_windows} windows -> sparse codes shape: {sparse_codes.shape}")
    print(f"  Average non-zero coefficients: {avg_nonzero:.1f} (target: {n_nonzero})")
    
    return sparse_codes


# ============================================================================
# PART 3: Signal Reconstruction (Decoding)
# ============================================================================

def reconstruct_signals(sparse_codes, dictionary):
    """
    Reconstruct EMG windows from their sparse codes.
    
    This is the DECOMPRESSION step. Given sparse codes X and dictionary D,
    the reconstructed signal is:
        Y_reconstructed = X @ D
    
    Each row of Y_reconstructed is a weighted sum of dictionary atoms.
    
    Parameters:
    -----------
    sparse_codes : np.ndarray, shape (n_windows, n_atoms)
        Sparse codes from encode_signals().
    dictionary : np.ndarray, shape (n_atoms, signal_dim)
        Dictionary atoms. Each row is one atom.
    
    Returns:
    --------
    X_reconstructed_flat : np.ndarray, shape (n_windows, signal_dim)
        Reconstructed signals in flattened form.
    """
    # Matrix multiplication: each row of sparse_codes selects and weights atoms
    X_reconstructed_flat = sparse_codes @ dictionary
    
    return X_reconstructed_flat


def reconstruct_to_original_shape(X_reconstructed_flat, original_shape):
    """
    Reshape reconstructed flat signals back to 3D window shape.
    
    Parameters:
    -----------
    X_reconstructed_flat : np.ndarray, shape (n_windows, signal_dim)
        Reconstructed signals from reconstruct_signals().
    original_shape : tuple
        Shape of the original X_windows: (n_windows, window_samples, n_channels)
    
    Returns:
    --------
    X_reconstructed : np.ndarray, same shape as original X_windows
    """
    n_windows, window_samples, n_channels = original_shape
    return X_reconstructed_flat.reshape(n_windows, window_samples, n_channels)


# ============================================================================
# PART 4: Complete Encode-Decode Pipeline
# ============================================================================

def compute_reconstruction_metrics(original, reconstructed):
    """
    Compute quality metrics for signal reconstruction.
    
    Parameters:
    -----------
    original : np.ndarray
        Original EMG windows (any shape, will be flattened for computation).
    reconstructed : np.ndarray
        Reconstructed EMG windows (same shape as original).
    
    Returns:
    --------
    metrics : dict
        Dictionary containing:
        - 'mse': Mean Squared Error (lower is better)
        - 'rmse': Root Mean Squared Error (lower is better, same units as signal)
        - 'mae': Mean Absolute Error (lower is better)
        - 'r2': R² score (closer to 1 is better, negative means worse than mean)
        - 'snr': Signal-to-Noise Ratio in dB (higher is better)
        - 'prd': Percentage Residual Difference (lower is better, as percentage)
    """
    # Flatten to 1D for computation
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Mean Squared Error
    mse = np.mean((orig_flat - recon_flat) ** 2)
    
    # Root Mean Squared Error (same units as the signal)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(orig_flat - recon_flat))
    
    # R² score (coefficient of determination)
    # R² = 1 means perfect reconstruction
    # R² = 0 means reconstruction is as good as the mean
    # R² < 0 means reconstruction is worse than just predicting the mean
    ss_res = np.sum((orig_flat - recon_flat) ** 2)
    ss_tot = np.sum((orig_flat - np.mean(orig_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Signal-to-Noise Ratio (dB)
    # SNR = 10 * log10(signal_power / noise_power)
    signal_power = np.mean(orig_flat ** 2)
    noise_power = mse
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = np.inf
    
    # Percentage Residual Difference
    # PRD = sqrt( sum((orig - recon)²) / sum(orig²) ) * 100
    sum_sq_orig = np.sum(orig_flat ** 2)
    if sum_sq_orig > 0:
        prd = np.sqrt(ss_res / sum_sq_orig) * 100
    else:
        prd = np.inf
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'snr': snr,
        'prd': prd
    }


def full_reconstruction_pipeline(X_windows, dict_learner, dictionary, n_nonzero=5):
    """
    Complete encode-decode pipeline: compress and reconstruct signals.
    
    This function:
    1. Encodes (compresses) the signals into sparse codes
    2. Decodes (reconstructs) the signals from sparse codes
    3. Computes quality metrics
    
    Parameters:
    -----------
    X_windows : np.ndarray, shape (n_windows, window_samples, n_channels)
        Original EMG windows.
    dict_learner : trained dictionary object
        For encoding (sparse coding).
    dictionary : np.ndarray, shape (n_atoms, signal_dim)
        Dictionary atoms for decoding (reconstruction).
    n_nonzero : int
        Sparsity level for encoding.
    
    Returns:
    --------
    X_reconstructed : np.ndarray, same shape as X_windows
        Reconstructed EMG windows.
    sparse_codes : np.ndarray, shape (n_windows, n_atoms)
        Sparse codes (the compressed representation).
    metrics : dict
        Reconstruction quality metrics.
    """
    # Step 1: Encode (compress)
    print("Step 1: Encoding signals into sparse codes...")
    sparse_codes = encode_signals(X_windows, dict_learner, n_nonzero)
    
    # Step 2: Decode (reconstruct)
    print("Step 2: Reconstructing signals from sparse codes...")
    X_flat_reconstructed = reconstruct_signals(sparse_codes, dictionary)
    
    # Step 3: Reshape back to 3D
    X_reconstructed = reconstruct_to_original_shape(
        X_flat_reconstructed, X_windows.shape
    )
    
    # Step 4: Compute metrics
    print("Step 3: Computing reconstruction metrics...")
    metrics = compute_reconstruction_metrics(X_windows, X_reconstructed)
    
    return X_reconstructed, sparse_codes, metrics


# ============================================================================
# PART 5: Per-Class Dictionaries (for Classification)
# ============================================================================

def train_per_class_dictionaries(X_train, y_train, n_atoms_per_class=30, alpha=0.1):
    """
    Train one dictionary per gesture class for SRC classification.
    
    This is a SEPARATE pipeline from the single-dictionary reconstruction.
    Each gesture gets its own specialized dictionary.
    
    Parameters:
    -----------
    X_train : np.ndarray, shape (n_windows, window_samples, n_channels)
    y_train : np.ndarray, shape (n_windows,)
    n_atoms_per_class : int
    alpha : float
    
    Returns:
    --------
    per_class_dicts : dict mapping gesture_id -> trained dictionary object
    """
    gesture_ids = sorted(np.unique(y_train))
    per_class_dicts = {}
    
    for gid in gesture_ids:
        mask = y_train == gid
        X_class = X_train[mask]
        n_windows = X_class.shape[0]
        
        X_flat = X_class.reshape(n_windows, -1)
        
        # Normalize to unit norm
        norms = np.linalg.norm(X_flat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_flat = X_flat / norms
        
        n_atoms_actual = min(n_atoms_per_class, max(5, n_windows // 3))
        
        print(f"  Gesture {gid:2d}: {n_windows:4d} windows -> {n_atoms_actual:3d} atoms")
        
        dico = MiniBatchDictionaryLearning(
            n_components=n_atoms_actual,
            alpha=alpha,
            batch_size=32,
            max_iter=500,
            random_state=42,
            transform_algorithm='omp'
        )
        dico.fit(X_flat)
        per_class_dicts[gid] = dico
    
    return per_class_dicts


def classify_by_reconstruction(X_test, per_class_dicts, n_nonzero=5):
    """
    Classify by minimum reconstruction error (SRC).
    
    For each test window, encode with each class dictionary and pick the one
    that reconstructs it best.
    """
    n_test = X_test.shape[0]
    class_ids = sorted(per_class_dicts.keys())
    
    y_pred = np.zeros(n_test, dtype=int)
    
    for i in range(n_test):
        window_flat = X_test[i].flatten()
        norm = np.linalg.norm(window_flat)
        if norm > 0:
            window_flat = window_flat / norm
        
        best_class = class_ids[0]
        best_error = np.inf
        
        for gid in class_ids:
            dico = per_class_dicts[gid]
            omp = OrthogonalMatchingPursuit(
                n_nonzero_coefs=n_nonzero, fit_intercept=False
            )
            omp.fit(dico.components_.T, window_flat)
            coeffs = omp.coef_
            reconstruction = dico.components_.T @ coeffs
            error = np.mean((window_flat - reconstruction) ** 2)
            
            if error < best_error:
                best_error = error
                best_class = gid
        
        y_pred[i] = best_class
        
        if (i + 1) % 500 == 0:
            print(f"  Classified {i+1}/{n_test} windows...")
    
    return y_pred