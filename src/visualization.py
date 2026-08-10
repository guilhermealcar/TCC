"""
Visualization functions for EMG analysis, classification, and reconstruction.

All functions return matplotlib Figure objects for flexibility.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# ============================================================================
# DICTIONARY VISUALIZATION
# ============================================================================

def plot_dictionary_atoms(dictionary, n_atoms=16, title="Dictionary Atoms"):
    """
    Visualize dictionary atoms as 1D waveform patterns.
    
    Each atom is a basic signal pattern learned from the data.
    Any EMG window can be expressed as a weighted sum of a few atoms.
    
    Parameters:
    -----------
    dictionary : np.ndarray, shape (n_atoms, signal_dim)
        The learned dictionary atoms.
    n_atoms : int
        Number of atoms to display (randomly selected).
    title : str
        Plot title.
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    np.random.seed(42)
    n_available = min(n_atoms, dictionary.shape[0])
    idx = np.random.choice(dictionary.shape[0], n_available, replace=False)
    atoms = dictionary[idx]
    
    cols = 4
    rows = int(np.ceil(n_available / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    
    # Handle case with only one subplot
    if n_available == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_available):
        axes[i].plot(atoms[i], color='teal', linewidth=1.5)
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[i].set_title(f'Atom {idx[i]}', fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Hide unused subplots
    for j in range(n_available, len(axes)):
        axes[j].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# CLASSIFICATION VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """
    Plot a confusion matrix as a heatmap.
    
    Diagonal elements show correct classifications.
    Off-diagonal elements show confusions between gestures.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True gesture labels.
    y_pred : np.ndarray
        Predicted gesture labels.
    class_names : list of str or None
        Names for each class. If None, uses the unique values in y_true.
    title : str
        Plot title.
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate class names if not provided
    if class_names is None:
        unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        class_names = [str(c) for c in unique_classes]
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.4), 
                                     max(6, len(class_names) * 0.35)))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names,
        ax=ax, 
        cbar_kws={'label': 'Count'},
        annot_kws={'size': max(6, 12 - len(class_names) // 5)}
    )
    
    ax.set_xlabel('Predicted Gesture', fontsize=12)
    ax.set_ylabel('True Gesture', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_accuracy_comparison(results_dict, title="Classification Accuracy Comparison"):
    """
    Bar chart comparing classification accuracy across methods.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with method names as keys and dicts containing 'accuracy' as values.
    title : str
        Plot title.
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    # Filter out None values
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    
    method_names = []
    accuracies = []
    
    for name, result in valid_results.items():
        # Format method name for display
        display_name = name.replace('_', ' ').title()
        method_names.append(display_name)
        accuracies.append(result['accuracy'] * 100)  # Convert to percentage
    
    # Find best accuracy
    best_acc = max(accuracies) if accuracies else 0
    colors = ['#1565C0' if a == best_acc else '#90CAF9' for a in accuracies]
    
    fig, ax = plt.subplots(figsize=(max(8, len(method_names) * 2), 5))
    bars = ax.bar(method_names, accuracies, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add accuracy labels on top of bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, max(accuracies) * 1.15 if accuracies else 100)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


# ============================================================================
# RECONSTRUCTION VISUALIZATION
# ============================================================================

def plot_reconstruction_comparison(original, reconstructed, method_name,
                                    n_examples=4, n_channels_to_show=4):
    """
    Plot original vs reconstructed EMG signals for visual comparison.
    
    Original is plotted in blue, reconstruction in red dashed.
    Shows multiple channels for multiple example windows.
    
    Parameters:
    -----------
    original : np.ndarray, shape (n_windows, window_samples, n_channels)
        Original EMG windows.
    reconstructed : np.ndarray, same shape
        Reconstructed EMG windows.
    method_name : str
        Name of the reconstruction method (for title).
    n_examples : int
        Number of example windows to show.
    n_channels_to_show : int
        Number of channels to display per example.
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    n_channels_actual = min(n_channels_to_show, original.shape[2])
    n_examples_actual = min(n_examples, original.shape[0])
    
    fig, axes = plt.subplots(n_examples_actual, n_channels_actual,
                              figsize=(4 * n_channels_actual, 3 * n_examples_actual))
    
    # Handle single subplot case
    if n_examples_actual == 1 and n_channels_actual == 1:
        axes = np.array([[axes]])
    elif n_examples_actual == 1:
        axes = axes.reshape(1, -1)
    elif n_channels_actual == 1:
        axes = axes.reshape(-1, 1)
    
    # Select random examples
    np.random.seed(42)
    example_indices = np.random.choice(original.shape[0], n_examples_actual, replace=False)
    
    for row, win_idx in enumerate(example_indices):
        for col in range(n_channels_actual):
            ax = axes[row, col]
            
            orig_signal = original[win_idx, :, col]
            recon_signal = reconstructed[win_idx, :, col]
            
            ax.plot(orig_signal, 'b-', label='Original', linewidth=1.5, alpha=0.8)
            ax.plot(recon_signal, 'r--', label='Reconstructed', linewidth=1.5, alpha=0.8)
            ax.set_title(f'Win {win_idx}, Ch {col+1}', fontsize=9)
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')
    
    fig.suptitle(f'Original vs Reconstructed EMG — {method_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_reconstruction_metrics_comparison(results_dict):
    """
    Bar chart comparing reconstruction metrics across methods.
    
    Shows RMSE (lower better), R² (higher better), and SNR (higher better).
    
    Parameters:
    -----------
    results_dict : dict
        Output of run_all_reconstruction_experiments().
        Keys are method names, values are dicts with 'metrics' key.
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    # Filter out None values
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    
    methods = []
    rmse_values = []
    r2_values = []
    snr_values = []
    
    for method_name, result in valid_results.items():
        display_name = method_name.replace('_', ' ').title()
        methods.append(display_name)
        rmse_values.append(result['metrics']['rmse'])
        r2_values.append(result['metrics']['r2'])
        snr_values.append(result['metrics']['snr'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE (lower is better)
    best_rmse = min(rmse_values)
    colors_rmse = ['#2ecc71' if v == best_rmse else '#bdc3c7' for v in rmse_values]
    axes[0].bar(methods, rmse_values, color=colors_rmse, edgecolor='white', linewidth=1.5)
    axes[0].set_title('RMSE\n(lower is better)', fontsize=12)
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val in zip(axes[0].patches, rmse_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.02,
                    f'{val:.4f}', ha='center', fontsize=9)
    
    # R² (higher is better, closer to 1)
    best_r2 = max(r2_values)
    colors_r2 = ['#2ecc71' if v == best_r2 else '#bdc3c7' for v in r2_values]
    axes[1].bar(methods, r2_values, color=colors_r2, edgecolor='white', linewidth=1.5)
    axes[1].set_title('R² Score\n(closer to 1 is better)', fontsize=12)
    axes[1].set_ylabel('R²')
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val in zip(axes[1].patches, r2_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', fontsize=9)
    
    # SNR (higher is better)
    best_snr = max(snr_values)
    colors_snr = ['#2ecc71' if v == best_snr else '#bdc3c7' for v in snr_values]
    axes[2].bar(methods, snr_values, color=colors_snr, edgecolor='white', linewidth=1.5)
    axes[2].set_title('SNR (dB)\n(higher is better)', fontsize=12)
    axes[2].set_ylabel('SNR (dB)')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val in zip(axes[2].patches, snr_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(snr_values)*0.02,
                    f'{val:.1f}', ha='center', fontsize=9)
    
    fig.suptitle('Reconstruction Quality Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# SPARSE CODE VISUALIZATION
# ============================================================================

def plot_sparse_code_heatmap(sparse_codes, title="Sparse Codes Heatmap"):
    """
    Visualize sparse codes as a heatmap to show sparsity patterns.
    
    White/light = near zero, colored = non-zero coefficients.
    A good sparse code should have mostly white entries (zeros).
    
    Parameters:
    -----------
    sparse_codes : np.ndarray, shape (n_windows, n_atoms)
        Sparse codes from dictionary encoding.
    title : str
        Plot title.
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    n_windows_to_show = min(50, sparse_codes.shape[0])
    
    fig, ax = plt.subplots(figsize=(12, max(6, n_windows_to_show * 0.12)))
    
    # Transpose so atoms are rows, windows are columns
    sns.heatmap(
        sparse_codes[:n_windows_to_show].T,
        cmap='RdBu_r', 
        center=0,
        xticklabels=5,  # Show every 5th label
        yticklabels=max(1, sparse_codes.shape[1] // 20),  # Show every Nth label
        ax=ax, 
        cbar_kws={'label': 'Coefficient Value'}
    )
    
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Atom Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================================
# RAW SIGNAL VISUALIZATION
# ============================================================================

def plot_signal_head(df_emg, labels=None, n_seconds=3, sampling_rate=200,
                      n_channels_to_show=5):
    """
    Visualize the first few seconds of EMG signal.
    
    Parameters:
    -----------
    df_emg : pd.DataFrame or np.ndarray, shape (n_samples, n_channels)
        EMG signal data.
    labels : np.ndarray or None
        Restimulus labels (0 = rest, >0 = gesture active).
    n_seconds : int
        Number of seconds to display.
    sampling_rate : int
        Sampling frequency in Hz.
    n_channels_to_show : int
        Number of channels to display.
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    limit = int(n_seconds * sampling_rate)
    
    # Convert to DataFrame if numpy array
    if isinstance(df_emg, np.ndarray):
        df_emg = pd.DataFrame(df_emg)
    
    n_channels_actual = min(n_channels_to_show, df_emg.shape[1])
    data_slice = df_emg.iloc[:limit, :n_channels_actual]
    time_axis = np.linspace(0, n_seconds, limit)
    
    fig, axes = plt.subplots(nrows=n_channels_actual, ncols=1,
                              figsize=(14, 1.8 * n_channels_actual), sharex=True)
    
    if n_channels_actual == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(time_axis, data_slice.iloc[:, i], color='steelblue', lw=1)
        ax.set_ylabel(f'Ch {i+1}', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Highlight active gesture regions
        if labels is not None:
            labels_flat = np.array(labels).flatten()
            active_zones = labels_flat[:limit] > 0
            
            # Find contiguous active regions for shading
            y_min, y_max = ax.get_ylim()
            ax.fill_between(time_axis, y_min, y_max,
                           where=active_zones, color='orange', alpha=0.15)
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    fig.suptitle(f'EMG Signal — First {n_seconds} seconds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_signal_and_envelope(df_signal, df_envelope, channel_idx=0, 
                              n_seconds=3, sampling_rate=200):
    """
    Plot a signal overlaid with its envelope.
    
    Parameters:
    -----------
    df_signal : pd.DataFrame or np.ndarray
        Rectified/filtered signal.
    df_envelope : pd.DataFrame or np.ndarray
        Low-pass filtered envelope.
    channel_idx : int
        Which channel to plot (0-indexed).
    n_seconds : int
        Number of seconds to display.
    sampling_rate : int
        Sampling frequency in Hz.
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    limit = int(n_seconds * sampling_rate)
    
    # Convert to numpy for consistent indexing
    if isinstance(df_signal, pd.DataFrame):
        signal = df_signal.values
    else:
        signal = df_signal
    
    if isinstance(df_envelope, pd.DataFrame):
        envelope = df_envelope.values
    else:
        envelope = df_envelope
    
    signal_slice = signal[:limit, channel_idx]
    envelope_slice = envelope[:limit, channel_idx]
    time_axis = np.linspace(0, n_seconds, limit)
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    ax.plot(time_axis, signal_slice, color='steelblue', alpha=0.5, 
            label='Rectified Signal', linewidth=1)
    ax.plot(time_axis, envelope_slice, color='red', linewidth=2, 
            label='Linear Envelope')
    
    ax.set_title(f'Signal vs Envelope — Channel {channel_idx + 1}', fontsize=14)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_signal_and_rms(df_signal, df_rms, channel_idx=0,
                         n_seconds=3, sampling_rate=200):
    """
    Plot a signal overlaid with its RMS continuous calculation.
    
    Parameters:
    -----------
    df_signal : pd.DataFrame or np.ndarray
        Rectified/filtered signal.
    df_rms : pd.DataFrame or np.ndarray
        RMS continuous values.
    channel_idx : int
        Which channel to plot (0-indexed).
    n_seconds : int
        Number of seconds to display.
    sampling_rate : int
        Sampling frequency in Hz.
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    limit = int(n_seconds * sampling_rate)
    
    if isinstance(df_signal, pd.DataFrame):
        signal = df_signal.values
    else:
        signal = df_signal
    
    if isinstance(df_rms, pd.DataFrame):
        rms = df_rms.values
    else:
        rms = df_rms
    
    signal_slice = signal[:limit, channel_idx]
    rms_slice = rms[:limit, channel_idx]
    time_axis = np.linspace(0, n_seconds, limit)
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    ax.plot(time_axis, signal_slice, color='steelblue', alpha=0.5,
            label='Rectified Signal', linewidth=1)
    ax.plot(time_axis, rms_slice, color='green', linewidth=2,
            label='RMS Continuous')
    
    ax.set_title(f'Signal vs RMS — Channel {channel_idx + 1}', fontsize=14)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


# ============================================================================
# COMPARATIVE CONFUSION MATRICES (for your original 04_Classification)
# ============================================================================

def plot_comparative_confusion_matrices(y_true, y_pred_class, y_pred_mb, y_pred_ksvd):
    """
    Plot three confusion matrices side by side for direct comparison.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels (same for all methods).
    y_pred_class : np.ndarray
        Predictions from classical features.
    y_pred_mb : np.ndarray
        Predictions from MiniBatch DL.
    y_pred_ksvd : np.ndarray
        Predictions from K-SVD.
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Confusion Matrices — Method Comparison', 
                 fontsize=18, fontweight='bold')
    
    preds = [y_pred_class, y_pred_mb, y_pred_ksvd]
    titles = ['Baseline (Classical Features)', 
              'MiniBatch Dictionary Learning', 
              'K-SVD Dictionary Learning']
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_true] + preds))
    
    for i, (ax, pred, title) in enumerate(zip(axes, preds, titles)):
        cm = confusion_matrix(y_true, pred, labels=classes)
        sns.heatmap(cm, annot=False, cmap='Blues', ax=ax, cbar=False)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig