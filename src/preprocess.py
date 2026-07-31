import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def apply_bandpass_filter(signal_data, lowcut=20.0, highcut=499.0, fs=1000.0, order=4):
    """
    Aplica um filtro Butterworth passa-banda para limpar o sEMG.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # filtfilt aplica o filtro indo e voltando, evitando atraso (phase shift) no sinal
    filtered_signal = filtfilt(b, a, signal_data, axis=0)
    return filtered_signal

def rectify_signal(signal_data):
    """
    Retificação de onda completa (Full-wave rectification).
    Rebate os valores negativos para o eixo positivo para focar na amplitude da energia.
    """
    return np.abs(signal_data)

def apply_lowpass_filter(signal_data, cutoff=5.0, fs=200.0, order=4):
    """
    Aplica um filtro Butterworth passa-baixa para extrair o envelope do sinal.
    Isso suaviza o sinal retificado, revelando apenas a intenção de movimento.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # btype='low' garante que apenas frequências menores que o cutoff passem
    b, a = butter(order, normal_cutoff, btype='low')
    
    filtered_signal = filtfilt(b, a, signal_data, axis=0)
    return filtered_signal

def calculate_rms(signal_data, window_ms=200, fs=200.0):
    """
    Calcula o RMS contínuo usando uma janela deslizante.
    Retorna a potência do sinal, sendo uma alternativa ao Envelope Linear.
    """
    window_samples = int((window_ms / 1000.0) * fs)
    
    if isinstance(signal_data, np.ndarray):
        signal_data = pd.DataFrame(signal_data)
        
    # Applica RMS: Raiz da Média dos Quadrados
    rms = np.sqrt(signal_data.pow(2).rolling(window=window_samples, min_periods=1, center=True).mean())
    
    # Preenche os NaNs das bordas copiando os valores mais próximos (backfill/forwardfill)
    rms = rms.bfill().ffill()
    return rms.values

def normalize_signal(signal_data, max_values=None):
    """
    Normalizes the signal (0 to 100%) based on the maximum values of each channel.
    """
    if max_values is None:
        max_values = np.max(signal_data, axis=0)
    
    max_values = np.where(max_values == 0, 1, max_values)  # Avoid division by zero
    
    normalized_signal = (signal_data / max_values) * 100.0
    return normalized_signal, max_values

def sliding_window(signal_data, labels, window_ms=200, overlap_pct=0.5, fs=200.0):
    """
    Independent from NinaPro dataset
    """
    window_samples = int((window_ms / 1000.0) * fs)
    step_samples = int(window_samples * (1.0 - overlap_pct))

    X_windows = []
    y_windows = []

    total_samples = signal_data.shape[0]

    for start in range(0, total_samples - window_samples + 1, step_samples):
        end = start + window_samples

        # Extract the 2D matrix corresponding to the current window time
        window = signal_data[start:end, :]
        X_windows.append(window)

        window_labels = labels[start:end]
        values, counts = np.unique(window_labels, return_counts=True)
        majority_label = values[np.argmax(counts)]
        y_windows.append(majority_label)

    return np.array(X_windows), np.array(y_windows)