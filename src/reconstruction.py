"""
Signal Reconstruction module.

This module handles the ENCODE-DECODE pipeline:
1. Encode: Compress EMG windows into low-dimensional sparse codes
2. Decode: Reconstruct the original EMG windows from sparse codes
3. Compare: Measure reconstruction quality across methods

Three methods are compared:
- MiniBatch Dictionary Learning
- K-SVD Dictionary Learning
- Classical Features (pseudo-inverse reconstruction, for comparison)
"""

import numpy as np
from src.dictionary import (
    train_minibatch_dictionary,
    train_ksvd_dictionary,
    full_reconstruction_pipeline,
    compute_reconstruction_metrics
)
from src.features import extract_features_batch, classical_features_to_signal


def run_all_reconstruction_experiments(X_train, X_test,
                                        n_atoms=100, n_nonzero=5,
                                        ksvd_max_iter=10):
    """
    Run reconstruction experiments comparing all three methods.
    
    For each method:
    1. Train the model on X_train
    2. Encode X_test into low-dimensional representation
    3. Decode back to original signal dimensions
    4. Compute reconstruction quality metrics
    
    Parameters:
    -----------
    X_train : np.ndarray, shape (n_windows, window_samples, n_channels)
        Training windows (used to learn dictionaries/transformations).
    X_test : np.ndarray, shape (n_windows, window_samples, n_channels)
        Test windows (used to evaluate reconstruction quality).
    n_atoms : int
        Number of dictionary atoms (compression dimension for DL methods).
    n_nonzero : int
        Sparsity level (number of non-zero coefficients).
    ksvd_max_iter : int
        Maximum K-SVD iterations.
    
    Returns:
    --------
    results : dict
        Dictionary with keys 'minibatch', 'ksvd', 'classical'.
        Each contains 'reconstructed', 'sparse_codes', 'metrics'.
    
    The results dictionary structure:
    {
        'minibatch': {
            'reconstructed': np.ndarray,  # Reconstructed test windows
            'sparse_codes': np.ndarray,   # Sparse codes (compressed representation)
            'metrics': dict,              # Quality metrics
            'compression_ratio': float,   # Original dimension / compressed dimension
        },
        'ksvd': { ... },
        'classical': {
            'reconstructed': np.ndarray,
            'features': np.ndarray,       # Classical features
            'metrics': dict,
            'compression_ratio': float,
        }
    }
    """
    results = {}
    
    # Get dimensions
    n_windows, window_samples, n_channels = X_train.shape
    signal_dim = window_samples * n_channels
    
    print(f"\nSignal dimension: {signal_dim} ({window_samples} samples × {n_channels} channels)")
    print(f"Compression target: {n_atoms} atoms (ratio {signal_dim / n_atoms:.1f}:1)")
    print(f"Sparsity: k = {n_nonzero}")
    
    # ========================================================================
    # Method 1: MiniBatch Dictionary Learning
    # ========================================================================
    print("\n" + "="*60)
    print("RECONSTRUCTION METHOD 1: MiniBatch Dictionary Learning")
    print("="*60)
    
    # Train dictionary on training data
    dict_mb, atoms_mb = train_minibatch_dictionary(
        X_train,
        n_components=n_atoms,
        alpha=1.0,
        max_iter=500
    )
    
    # Reconstruct test data
    X_recon_mb, sparse_codes_mb, metrics_mb = full_reconstruction_pipeline(
        X_test, dict_mb, atoms_mb, n_nonzero
    )
    
    results['minibatch'] = {
        'reconstructed': X_recon_mb,
        'sparse_codes': sparse_codes_mb,
        'metrics': metrics_mb,
        'compression_ratio': signal_dim / n_atoms
    }
    
    print(f"  MiniBatch DL Results:")
    print(f"    RMSE: {metrics_mb['rmse']:.6f}")
    print(f"    R²:   {metrics_mb['r2']:.4f}")
    print(f"    SNR:  {metrics_mb['snr']:.2f} dB")
    
    # ========================================================================
    # Method 2: K-SVD Dictionary Learning
    # ========================================================================
    print("\n" + "="*60)
    print("RECONSTRUCTION METHOD 2: K-SVD Dictionary Learning")
    print("="*60)
    
    try:
        dict_ksvd, atoms_ksvd = train_ksvd_dictionary(
            X_train,
            n_components=n_atoms,
            transform_n_nonzero_coefs=n_nonzero,
            max_iter=ksvd_max_iter
        )
        
        X_recon_ksvd, sparse_codes_ksvd, metrics_ksvd = full_reconstruction_pipeline(
            X_test, dict_ksvd, atoms_ksvd, n_nonzero
        )
        
        results['ksvd'] = {
            'reconstructed': X_recon_ksvd,
            'sparse_codes': sparse_codes_ksvd,
            'metrics': metrics_ksvd,
            'compression_ratio': signal_dim / n_atoms
        }
        
        print(f"  K-SVD DL Results:")
        print(f"    RMSE: {metrics_ksvd['rmse']:.6f}")
        print(f"    R²:   {metrics_ksvd['r2']:.4f}")
        print(f"    SNR:  {metrics_ksvd['snr']:.2f} dB")
        
    except ImportError:
        print("  K-SVD not available (ksvd package not installed). Skipping.")
        results['ksvd'] = None
    
    # ========================================================================
    # Method 3: Classical Features (pseudo-inverse reconstruction)
    # ========================================================================
    print("\n" + "="*60)
    print("RECONSTRUCTION METHOD 3: Classical Features (Pseudo-Inverse)")
    print("="*60)
    
    # Extract classical features
    classical_features = extract_features_batch(X_test)
    n_classical_features = classical_features.shape[1]
    
    # Attempt reconstruction using pseudo-inverse
    X_recon_classical = classical_features_to_signal(
        classical_features, n_channels, window_samples
    )
    
    metrics_classical = compute_reconstruction_metrics(X_test, X_recon_classical)
    
    results['classical'] = {
        'reconstructed': X_recon_classical,
        'features': classical_features,
        'metrics': metrics_classical,
        'compression_ratio': signal_dim / n_classical_features
    }
    
    print(f"  Classical Features Results:")
    print(f"    RMSE: {metrics_classical['rmse']:.6f}")
    print(f"    R²:   {metrics_classical['r2']:.4f}")
    print(f"    SNR:  {metrics_classical['snr']:.2f} dB")
    print(f"  Note: Classical features lose temporal information (ZC, SSC are non-linear).")
    print(f"        The pseudo-inverse reconstruction is an APPROXIMATION.")
    
    # ========================================================================
    # Summary Comparison
    # ========================================================================
    print("\n" + "="*60)
    print("RECONSTRUCTION COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Method':<30} {'RMSE':>10} {'R²':>10} {'SNR (dB)':>10} {'Comp Ratio':>12}")
    print("-"*72)
    
    print(f"{'MiniBatch DL':<30} {metrics_mb['rmse']:>10.6f} {metrics_mb['r2']:>10.4f} {metrics_mb['snr']:>10.2f} {signal_dim/n_atoms:>12.1f}:1")
    
    if results['ksvd'] is not None:
        print(f"{'K-SVD DL':<30} {metrics_ksvd['rmse']:>10.6f} {metrics_ksvd['r2']:>10.4f} {metrics_ksvd['snr']:>10.2f} {signal_dim/n_atoms:>12.1f}:1")
    
    print(f"{'Classical Features':<30} {metrics_classical['rmse']:>10.6f} {metrics_classical['r2']:>10.4f} {metrics_classical['snr']:>10.2f} {signal_dim/n_classical_features:>12.1f}:1")
    
    # Interpret results
    print("\n" + "-"*60)
    print("INTERPRETATION:")
    print(f"  • Higher R² (closer to 1) = better reconstruction")
    print(f"  • Higher SNR = better signal quality")
    print(f"  • Dictionary methods preserve temporal structure")
    print(f"  • Classical features lose temporal info (ZC/SSC are non-linear)")
    print(f"  • Compression ratio shows dimensionality reduction")
    
    return results