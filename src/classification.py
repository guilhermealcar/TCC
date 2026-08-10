"""
Classification module for EMG gesture recognition.

Compares three feature representations:
1. SRC Direct: Per-class dictionaries with minimum reconstruction error
2. Sparse Features + SVM: Sparse codes from single dictionary + SVM classifier
3. Classical Features + SVM: Time-domain features + SVM classifier (baseline)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dictionary import (
    train_minibatch_dictionary,
    encode_signals,
    train_per_class_dictionaries,
    classify_by_reconstruction
)
from src.features import extract_features_batch


def run_all_classification_experiments(X_train, X_test, y_train, y_test,
                                        n_atoms=100, n_nonzero=5):
    """
    Run all three classification methods and return results.
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray, shape (n_windows, window_samples, n_channels)
    y_train, y_test : np.ndarray, shape (n_windows,)
    n_atoms : int
        Number of dictionary atoms for single-dictionary methods.
    n_nonzero : int
        Sparsity level.
    
    Returns:
    --------
    results : dict
        Dictionary with keys 'src', 'sparse_svm', 'classical_svm'.
        Each contains 'accuracy', 'predictions', 'report'.
    """
    results = {}
    
    # ========================================================================
    # Method 1: SRC Direct (Per-Class Dictionaries)
    # ========================================================================
    print("\n" + "="*60)
    print("METHOD 1: SRC Direct (Per-Class Dictionaries)")
    print("="*60)
    
    per_class_dicts = train_per_class_dictionaries(
        X_train, y_train, n_atoms_per_class=30, alpha=0.1
    )
    y_pred_src = classify_by_reconstruction(X_test, per_class_dicts, n_nonzero=n_nonzero)
    
    acc_src = accuracy_score(y_test, y_pred_src)
    results['src'] = {
        'accuracy': acc_src,
        'predictions': y_pred_src,
        'report': classification_report(y_test, y_pred_src, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_src)
    }
    print(f"Accuracy: {acc_src * 100:.2f}%")
    
    # ========================================================================
    # Method 2: Sparse Features from Single Dictionary + SVM
    # ========================================================================
    print("\n" + "="*60)
    print("METHOD 2: Sparse Codes + SVM (Single Dictionary)")
    print("="*60)
    
    # Train a single dictionary on all training data
    dict_learner, _ = train_minibatch_dictionary(
        X_train, n_components=n_atoms, alpha=1.0, max_iter=500
    )
    
    # Encode both train and test
    X_train_sparse = encode_signals(X_train, dict_learner, n_nonzero)
    X_test_sparse = encode_signals(X_test, dict_learner, n_nonzero)
    
    # Normalize and classify
    scaler = StandardScaler()
    X_train_sparse_scaled = scaler.fit_transform(X_train_sparse)
    X_test_sparse_scaled = scaler.transform(X_test_sparse)
    
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train_sparse_scaled, y_train)
    y_pred_sparse = svm.predict(X_test_sparse_scaled)
    
    acc_sparse = accuracy_score(y_test, y_pred_sparse)
    results['sparse_svm'] = {
        'accuracy': acc_sparse,
        'predictions': y_pred_sparse,
        'report': classification_report(y_test, y_pred_sparse, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_sparse)
    }
    print(f"Accuracy: {acc_sparse * 100:.2f}%")
    
    # ========================================================================
    # Method 3: Classical Features + SVM (Baseline)
    # ========================================================================
    print("\n" + "="*60)
    print("METHOD 3: Classical TD Features + SVM (Baseline)")
    print("="*60)
    
    X_train_classical = extract_features_batch(X_train)
    X_test_classical = extract_features_batch(X_test)
    
    scaler_cls = StandardScaler()
    X_train_classical_scaled = scaler_cls.fit_transform(X_train_classical)
    X_test_classical_scaled = scaler_cls.transform(X_test_classical)
    
    svm_cls = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_cls.fit(X_train_classical_scaled, y_train)
    y_pred_classical = svm_cls.predict(X_test_classical_scaled)
    
    acc_classical = accuracy_score(y_test, y_pred_classical)
    results['classical_svm'] = {
        'accuracy': acc_classical,
        'predictions': y_pred_classical,
        'report': classification_report(y_test, y_pred_classical, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_classical)
    }
    print(f"Accuracy: {acc_classical * 100:.2f}%")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"SRC Direct................: {acc_src * 100:.2f}%")
    print(f"Sparse Codes + SVM........: {acc_sparse * 100:.2f}%")
    print(f"Classical Features + SVM..: {acc_classical * 100:.2f}%")
    
    return results