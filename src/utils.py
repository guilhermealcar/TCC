import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import zipfile
from scipy.signal import butter, filtfilt, welch
from sklearn.decomposition import sparse_encode
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

## DATA LOADING

def load_emg_data(zip_path):
    '''
    Loads all .mat file from the .zip files and returns it as a pandas DataFrame.
    Returns emg, stimulus, restimulus and repetition
    '''
    all_emg = []
    all_stimulus = []
    all_restimulus = []
    all_repetition = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for fname in sorted([f for f in zf.namelist() if f.endswith('.mat')]):
            with zf.open(fname) as f:
                mat = sio.loadmat(io.BytesIO(f.read()))
                all_emg.append(pd.DataFrame(mat['emg']))
                all_stimulus.append(mat['stimulus'])
                all_restimulus.append(mat['restimulus'])
                all_repetition.append(mat['repetition'])

    return (pd.concat(all_emg).reset_index(drop=True),
            np.concatenate(all_stimulus).flatten(),
            np.concatenate(all_restimulus).flatten(),
            np.concatenate(all_repetition).flatten())

## SIGNAL PROCESSING

def butter_bandpass(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # axis=0 to filter along the time axis (rows)
    filtered_data = filtfilt(b, a, data, axis=0) 
    return filtered_data

def preprocess(df, fs, lowcut, highcut, window_ms, overlap):
    """
    Filter -> rectify -> Z-score normalise -> sliding window.
    Returns Y: (n_windows, samples_per_window, n_channels)
    """
    emg = butter_bandpass(df.values, lowcut, highcut, fs)
    emg = np.abs(emg)
    emg = (emg - emg.mean(axis=0)) / (emg.std(axis=0) + 1e-8)

    win    = int(window_ms / 1000 * fs)
    step   = int(win * (1 - overlap))
    n_ch   = emg.shape[1]

    # Calcula índices válidos — só janelas com tamanho completo
    starts = np.arange(0, len(emg) - win, step)
    n_win  = len(starts)

    # Pré-aloca com shape explícito — nunca cria object array
    Y = np.empty((n_win, win, n_ch), dtype=np.float32)
    for k, i in enumerate(starts):
        Y[k] = emg[i:i + win, :]

    return Y   # (n_windows, samples_per_window, n_channels)

def sync_labels(signal, fs, window_ms, overlap, threshold=0.5):
    """
    Extracts the predominant label for each time window.
    """
    win = int(window_ms / 1000 * fs)
    step = int(win * (1 - overlap))
    
    labels = []
    for i in range(0, len(signal) - win, step):
        seg = signal[i : i + win].flatten().astype(int)
        counts = np.bincount(seg)
        most_frequent = np.argmax(counts)
        labels.append(most_frequent if counts[most_frequent] / len(seg) >= threshold else -1)  # -1 for rest if no dominant label

    return np.array(labels)

def sync_repetitions(rep, fs, window_ms, overlap):
    """
    Extracts the predominant repetition for each time window.
    """
    win = int(window_ms / 1000 * fs)
    step = int(win * (1 - overlap))
    
    repetitions = []
    for i in range(0, len(rep) - win, step):
        repetitions.append(np.bincount(rep[i : i + win].flatten().astype(int)).argmax())

    return np.array(repetitions)

def split_by_repetition(X, y, reps, test_rep=6):
    """
    X: windowed EMG data
    y: gesture labels
    reps: repetition labels
    test_rep: which repetition to use for testing
    
    Common protocol: train on reps 1,3,4,6 and test on reps 2,5
    Or simpler: test on rep 6, train on rest
    """
    # Training: all windows NOT from test_rep, AND gesture > 0 (not rest)
    train_mask = (reps != test_rep) & (y > 0)
    # Testing: all windows FROM test_rep, AND gesture > 0
    test_mask = (reps == test_rep) & (y > 0)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Train: {X_train.shape[0]} windows from reps {np.unique(reps[train_mask])}")
    print(f"Test:  {X_test.shape[0]} windows from rep {test_rep}")
    print(f"Train classes: {np.unique(y_train)}")
    print(f"Test classes:  {np.unique(y_test)}")
    
    return X_train, X_test, y_train, y_test

## FEATURE EXTRACTION

def to_per_channel(Y_3d):
    '''
    Reshapes Y from (n_windows, samples_per_window, n_channels) to (n_windows * n_channels, samples_per_window)
    '''
    n_windows, samples_per_window, n_channels = Y_3d.shape
    return Y_3d.transpose(0, 2, 1).reshape(n_windows * n_channels, samples_per_window)

def sparse_features(Y_3d, D, sparsity):
    '''
    Applies sparse coding to each window in Y using dictionary D.
    Returns a feature matrix of shape (n_windows, n_atoms).
    '''
    n_windows, samples_per_window, n_channels = Y_3d.shape
    Xch = sparse_encode(to_per_channel(Y_3d), D, algorithm='omp', n_nonzero_coefs=sparsity, n_jobs=-1)
    features = Xch.reshape(n_windows, n_channels, D.shape[0]).reshape(n_windows, n_channels * D.shape[0])
    
    return features

def handcrafted_features(Y_3d):
    '''
    Extracts classic features from sEMG data (MAV, RMS, WL) per channel
    Returns array of shape (n_windows, n_channels * 3)
    '''
    n_windows, samples_per_window, n_channels = Y_3d.shape
    features = []
    for i in range(n_windows):
        window = Y_3d[i]
        mav = np.mean(np.abs(window), axis=0)  # Mean Absolute Value
        rms = np.sqrt(np.mean(window**2, axis=0))  # Root Mean Square
        wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)  # Waveform Length
        features.append(np.concatenate([mav, rms, wl]))

    return np.array(features)  # shape: (n_windows, n_channels * 3)

def build_features(Y_train, Y_test, D, sparsity):
    X_train_sparse = sparse_features(Y_train, D, sparsity)
    X_test_sparse  = sparse_features(Y_test,  D, sparsity)
    X_train_hand   = handcrafted_features(Y_train)
    X_test_hand    = handcrafted_features(Y_test)

    sc1, sc2 = StandardScaler(), StandardScaler()
    X_train_sparse = sc1.fit_transform(X_train_sparse)
    X_test_sparse  = sc1.transform(X_test_sparse)
    X_train_hand   = sc2.fit_transform(X_train_hand)
    X_test_hand    = sc2.transform(X_test_hand)

    # Concatena e retorna 2 valores — (train, test)
    return (np.hstack([X_train_sparse, X_train_hand]),
            np.hstack([X_test_sparse,  X_test_hand]))

## CLASSIFICATION

CLASSIFIERS = {
    'Random Forest':       RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    'SVM (RBF)':           SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'LDA':                 LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
}

def run_classification(X_train, X_test, y_train, y_test, title=""):
    """
    Trains and evaluates all classifiers in CLASSIFIERS on the provided train/test data.
    """
    results = {}
    print(f"\n{'='*55}\n  {title}\n{'='*55}")
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}\n{'-'*55}")
    for name, clf in CLASSIFIERS.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'acc': acc,
            'cm':  confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, zero_division=0),
            'clf': clf,
        }
        print(f"  {name:<25}  {acc:.2%}")
    best = max(results, key=lambda k: results[k]['acc'])
    print(f"\n  Best: {best}  ({results[best]['acc']:.2%})")
    
    return results

## VISUALIZATION

def plot_raw_emg(df, stimulus, restimulus, fs, seconds, title=""):
    n = int(seconds * fs)
    t = np.arange(n) / fs
    n_ch = df.shape[1]

    fig, axes = plt.subplots(3, 1, figsize=(16, 9),
                              gridspec_kw={'height_ratios': [5, 1, 1]})

    # Normaliza cada canal para visualização (divide pelo max abs do canal)
    # Isso resolve a diferença de escala entre DB1 (mV) e DB2 (V)
    for ch in range(n_ch):
        sig = df.iloc[:n, ch].values          # CORRETO: linha × coluna
        sig_norm = sig / (np.abs(sig).max() + 1e-8)   # normaliza para [-1, 1]
        axes[0].plot(t, sig_norm + ch * 2.2,  # offset por canal
                     lw=0.7, alpha=0.85, label=f'Channel {ch+1}')

    axes[0].set_title(f'Raw EMG Signals - {title}')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Normalised amplitude (offset per channel)')
    axes[0].legend(loc='upper right', fontsize=6, ncol=2)

    # Stimulus — usa fill_between, não imshow (que distorce o eixo x)
    t_full = np.arange(len(stimulus)) / fs
    axes[1].fill_between(t_full[:n], stimulus[:n].flatten(),
                          alpha=0.8, color='steelblue')
    axes[1].set_title('Stimulus')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Gesture ID')

    # Restimulus
    axes[2].fill_between(t_full[:n], restimulus[:n].flatten(),
                          alpha=0.8, color='darkorange')
    axes[2].set_title('Restimulus')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Gesture ID')

    plt.tight_layout()
    plt.show()

def plot_psd(df, fs, n_channels, title=""):
    fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4), sharey=True)
    for i, ax in enumerate(axes):
        f, Pxx = welch(df.iloc[:, i].values, fs=fs, nperseg=min(512, len(df)//4))
        ax.semilogy(f, Pxx, color=f'C{i}')
        ax.set_title(f'Channel {i+1} PSD - {title}')
        ax.set_xlabel('Frequency (Hz)')
        if i == 0: ax.set_ylabel('PSD (V^2/Hz)')
        ax.grid(True, linestyle='--', alpha=0.4)
    fig.suptitle(f'{title} - Power Spectral Density', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_rms_heatmap(df, restim, n_channels, title=""):
    n = min(len(df), len(restim))
    df_t = df.iloc[:n, :n_channels].copy()

    # Converte para mesma ordem de grandeza (normaliza por canal)
    # Permite comparar padrões mesmo com unidades diferentes (V vs mV)
    for col in df_t.columns:
        mx = df_t[col].abs().max()
        if mx > 0:
            df_t[col] = df_t[col] / mx

    df_t['label'] = restim[:n].flatten()
    act = df_t.groupby('label').apply(
        lambda x: np.sqrt(np.mean(x.iloc[:, :n_channels].values**2, axis=0)))
    act = pd.DataFrame(act.tolist(), index=act.index,
                        columns=[f'Ch{i+1}' for i in range(n_channels)])

    plt.figure(figsize=(max(8, n_channels), min(20, len(act)*0.4+2)))
    sns.heatmap(act.iloc[1:], annot=True, fmt='.3f', cmap='Blues', linewidths=0.3)
    plt.title(f'{title} — Relative RMS activation per channel/gesture')
    plt.tight_layout()
    plt.show()

def plot_dictionary_atoms(D, n_atoms_to_show=20, title=""):
    """
    D: (n_atoms, samples). Each atom is a 1D waveform
    """
    cols = 5; rows = int(np.ceil(n_atoms_to_show/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*2.2))
    axes = axes.flatten()
    for i in range(n_atoms_to_show):
        axes[i].plot(D[i], color='steelblue', lw=1.2)
        axes[i].axhline(0, color='gray', lw=0.5, ls='--')
        axes[i].set_title(f'Atom {i}', fontsize=8); axes[i].tick_params(labelsize=6)
    for j in range(n_atoms_to_show, len(axes)): axes[j].axis('off')
    fig.suptitle(f'{title} — Dictionary atoms', fontsize=11)
    plt.tight_layout(); plt.show()

def plot_reconstruction_error(Y_3d, D, sparsity, n=300, title=""):
    Ych = to_per_channel(Y_3d[:n])
    Xch = sparse_encode(Ych, D, algorithm='omp', n_nonzero_coefs=sparsity, n_jobs=-1)
    err = np.linalg.norm(Ych - Xch @ D, axis=1)
    plt.figure(figsize=(9, 3))
    plt.hist(err, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    plt.axvline(err.mean(), color='red', ls='--', label=f'Mean {err.mean():.3f}')
    plt.title(f'{title} — Reconstruction error'); plt.xlabel('||y - Dx||'); plt.legend()
    plt.tight_layout(); plt.show()
    return err

def plot_sparse_heatmap(Y_3d, D, y, sparsity, n_atoms_to_show=60, title=""):
    n, s, c = Y_3d.shape
    Xch  = sparse_encode(to_per_channel(Y_3d), D,
                          algorithm='omp', n_nonzero_coefs=sparsity, n_jobs=-1)
    Xwin = Xch.reshape(n, c, D.shape[0]).mean(axis=1)
    df   = pd.DataFrame(Xwin[:, :min(n_atoms_to_show, D.shape[0])])
    df['label'] = y
    act  = df[df['label'] > 0].groupby('label').mean()
    plt.figure(figsize=(16, max(5, len(act)*0.35)))
    sns.heatmap(act, cmap='RdBu_r', center=0, xticklabels=10)
    plt.title(f'{title} — Mean sparse activation per gesture')
    plt.xlabel('Atom'); plt.ylabel('Gesture')
    plt.tight_layout(); plt.show()

def plot_accuracy_comparison(results, title=""):
    """
    Bar chart comparing all classifiers in results dict.
    """

    names = list(results.keys())
    accs  = [results[n]['acc']*100 for n in names]
    best  = max(accs)
    colors = ['#1565C0' if a == best else '#90CAF9' for a in accs]
    plt.figure(figsize=(10, 4))
    bars = plt.bar(names, accs, color=colors, edgecolor='white')
    for b, a in zip(bars, accs):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 f'{a:.1f}%', ha='center', fontsize=9)
    plt.ylim(0, 105); plt.ylabel('Accuracy (%)'); plt.title(f'{title} — Classifier comparison')
    plt.xticks(rotation=15, ha='right'); plt.tight_layout(); plt.show()

def plot_confusion_matrix(cm, labels, title=""):
    n = len(labels); sz = max(7, n*0.4)
    plt.figure(figsize=(sz, sz*0.85))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': max(5, 10-n//8)})
    plt.title(f'Confusion Matrix — {title}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.show()