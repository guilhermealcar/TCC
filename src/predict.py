# predict.py
import argparse
import os
import joblib
import numpy as np
from scipy.stats import mode
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from config import MODELS_DIR, MOVEMENT_LABELS
from preprocess import sEMGPreprocessor
from features import extract_fft_magnitude, extract_time_domain_features


def smooth_predictions(predictions: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Applies a sliding majority vote to smooth out jumpy predictions."""
    smoothed = np.copy(predictions)
    pad = window_size // 2
    for i in range(pad, len(predictions) - pad):
        window = predictions[i - pad : i + pad + 1]
        smoothed[i] = mode(window, keepdims=True)[0][0]
    return smoothed

def generate_comparison_log(y_true: np.ndarray, y_pred: np.ndarray, stride_ms: int = 10):
    """Prints a side-by-side comparison of the Ground Truth vs Predictions."""
    print("\n" + "=" * 85)
    print(" sEMG TRUE vs PREDICTED TIMELINE LOG (Hybrid DL + RF)")
    print("=" * 85)

    if len(y_true) == 0:
        return

    current_true = y_true[0]
    current_pred = y_pred[0]
    start_idx = 0

    for i in range(1, len(y_true)):
        if y_true[i] != current_true or y_pred[i] != current_pred:
            duration = (i - start_idx) * (stride_ms / 1000.0)
            
            if duration >= 0.2:
                start_time = (start_idx * stride_ms) / 1000.0
                end_time = (i * stride_ms) / 1000.0
                
                true_text = MOVEMENT_LABELS.get(current_true, "Unknown")
                pred_text = MOVEMENT_LABELS.get(current_pred, "Unknown")
                match = "✅" if current_true == current_pred else "❌"
                
                print(f"[{start_time:06.2f}s - {end_time:06.2f}s] {match} TRUE: {true_text:<28} | PRED: {pred_text}")
            
            current_true = y_true[i]
            current_pred = y_pred[i]
            start_idx = i
            
    print("=" * 85 + "\n")

def print_class_table(y_true: np.ndarray, y_pred: np.ndarray):
    """Prints a clean reference table of the movements detected in the file."""
    # Find all unique classes present in either the ground truth or predictions
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    
    print("\n" + "=" * 60)
    print(f" {'ID':<4} | {'MOVEMENT REFERENCE TABLE':<40}")
    print("=" * 60)
    for cls in unique_classes:
        label = MOVEMENT_LABELS.get(cls, "Unknown")
        print(f" {cls:<4} | {label}")
    print("=" * 60 + "\n")

def plot_signal_only(emg_mav: np.ndarray, stride_ms: int = 10):
    """Generates a single-pane plot of the raw physical muscle energy."""
    print("Generating signal graphic...")
    time_axis = np.arange(len(emg_mav)) * (stride_ms / 1000.0)
    
    plt.figure(figsize=(14, 4))
    
    plt.plot(time_axis, emg_mav, color='gray', alpha=0.8)
    plt.title("Raw sEMG Muscle Activation (Mean Absolute Value)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "signal_activation.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Graph saved successfully to '{plot_path}'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Predict hand movements using Hybrid DL + RF.")
    parser.add_argument("--file", type=str, required=True, help="Path to the raw NinaPro .mat file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return

    print(f"Loading raw signal from {args.file}...")
    mat_data = sio.loadmat(args.file)
    emg_raw = mat_data['emg']
    
    true_labels = mat_data['restimulus'].squeeze()
    offset = 0
    if 'E2' in args.file:
        offset = 12
    elif 'E3' in args.file:
        offset = 29
        
    active_mask = true_labels > 0
    true_labels[active_mask] += offset

    print("Preprocessing signal (Filtering & Standardizing)...")
    preprocessor = sEMGPreprocessor(sample_rate=100)
    filtered_emg = preprocessor.filter_signal(emg_raw)
    norm_emg = preprocessor.fit_standardize(filtered_emg)

    print("Extracting 200ms sliding windows...")
    X_windows, y_true_aligned, _ = preprocessor.extract_dense_windows(norm_emg, true_labels, true_labels)

    # 1. Frequency Domain Features & Sparse Coding
    print("Extracting FFT Frequencies...")
    X_freq = extract_fft_magnitude(X_windows)

    print("Loading Trained Models (Dictionary, Scaler, Random Forest)...")
    try:
        dict_learner = joblib.load(os.path.join(MODELS_DIR, "fft_dictionary_s1.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_hybrid_s1.pkl"))
        rf_classifier = joblib.load(os.path.join(MODELS_DIR, "rf_classifier_hybrid_s1.pkl"))
    except FileNotFoundError as e:
        print(f"Model error: {e}. Please ensure you ran notebook 03 first.")
        return

    print("Extracting Sparse Dictionary Codes...")
    X_flat = X_freq.reshape(X_freq.shape[0], -1)
    X_sparse = dict_learner.transform(X_flat)

    # 2. Time Domain Features
    print("Extracting Time-Domain Features...")
    X_td = extract_time_domain_features(X_windows)

    # 3. Hybrid Feature Fusion
    X_hybrid = np.hstack((X_sparse, X_td))

    print("Classifying movements with Random Forest...")
    X_scaled = scaler.transform(X_hybrid)
    raw_predictions = rf_classifier.predict(X_scaled)

    # Energy Gate
    window_energy = np.mean(np.abs(X_windows), axis=(1, 2))
    rest_threshold = 0.12 
    raw_predictions[window_energy < rest_threshold] = 0

    smoothed_preds = smooth_predictions(raw_predictions, window_size=50)

    # Comparison Log
    generate_comparison_log(y_true_aligned, smoothed_preds)

    # Accuracy Metrics
    overall_acc = accuracy_score(y_true_aligned, smoothed_preds)
    active_indices = y_true_aligned > 0
    active_acc = accuracy_score(y_true_aligned[active_indices], smoothed_preds[active_indices]) if np.any(active_indices) else 0.0

    print(f"[EVALUATION RESULTS - HYBRID RF]")
    print(f"Overall Accuracy:       {overall_acc * 100:.2f}%")
    print(f"Active Blocks Accuracy: {active_acc * 100:.2f}%")
    print("=" * 85 + "\n")
    
    # 1. Print the clean ID-to-Text reference table
    print_class_table(y_true_aligned, smoothed_preds)
    
    # 2. Plot ONLY the physical signal
    emg_mav = np.mean(np.abs(norm_emg), axis=1)
    emg_mav_aligned = emg_mav[19:] 
    
    plot_signal_only(emg_mav_aligned)

if __name__ == "__main__":
    main()