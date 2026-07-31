import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def signal_head(df_emg, labels=None, n_seconds=3, sampling_rate=200, n_channels_to_show=5):
    """
    Simula um df.head() visual e estrutural para sinais sEMG contínuos.
    """
    limit = int(n_seconds * sampling_rate)
    
    if isinstance(df_emg, np.ndarray):
        df_emg = pd.DataFrame(df_emg)
        
    data_slice = df_emg.iloc[:limit, :n_channels_to_show]
    time_axis = np.linspace(0, n_seconds, limit)
    
    print("=== Signal Structural Head ===")
    print(f"Total Shape : {df_emg.shape} (Samples x Channels)")
    print(f"Previewing  : First {n_seconds}s ({limit} samples) across {n_channels_to_show} channels.")
    print("-" * 40)
    print(data_slice.head())
    print("=" * 40)
    
    fig, axes = plt.subplots(nrows=n_channels_to_show, ncols=1, 
                             figsize=(12, 1.5 * n_channels_to_show), sharex=True)
    
    if n_channels_to_show == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        ax.plot(time_axis, data_slice.iloc[:, i], color='steelblue', lw=1)
        ax.set_ylabel(f'Ch {i+1}', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Sombreia o fundo de laranja se o paciente estiver realizando um movimento
        if labels is not None:
            labels = np.array(labels).flatten()
            active_zones = labels[:limit] > 0
            ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1], 
                            where=active_zones, color='orange', alpha=0.2)

    axes[-1].set_xlabel('Time (Seconds)', fontsize=12)
    plt.suptitle(f'Signal Visual Head - First {n_seconds}s', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_signal_and_envelope(df_signal, df_envelope, channel_idx=0, n_seconds=3, sampling_rate=200):
    """
    Plota o sinal (retificado ou filtrado) sobreposto pelo seu envelope linear.
    
    Parâmetros:
    -----------
    df_signal : pd.DataFrame ou np.ndarray (Sinal base, geralmente retificado)
    df_envelope : pd.DataFrame ou np.ndarray (Envelope gerado pelo filtro passa-baixa)
    channel_idx : int (Índice do canal a ser visualizado. 0 = Canal 1)
    n_seconds : int (Segundos iniciais para plotar)
    sampling_rate : int (Frequência de amostragem)
    """
    limit = int(n_seconds * sampling_rate)
    
    # Converte para DataFrame se for ndarray para facilitar o slicing
    if isinstance(df_signal, np.ndarray):
        df_signal = pd.DataFrame(df_signal)
    if isinstance(df_envelope, np.ndarray):
        df_envelope = pd.DataFrame(df_envelope)
        
    signal_slice = df_signal.iloc[:limit, channel_idx]
    envelope_slice = df_envelope.iloc[:limit, channel_idx]
    time_axis = np.linspace(0, n_seconds, limit)
    
    plt.figure(figsize=(12, 4))
    
    # Plota o sinal retificado em azul com transparência (alpha) para não ofuscar o envelope
    plt.plot(time_axis, signal_slice, color='steelblue', alpha=0.5, label='Rectified Signal')
    
    # Plota o envelope em vermelho e mais espesso
    plt.plot(time_axis, envelope_slice, color='red', linewidth=2, label='Linear Envelope')
    
    plt.title(f'Superposition: Rectified Signal vs Envelope (Channel {channel_idx + 1})', fontsize=14)
    plt.xlabel('Time (Seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(loc='upper right')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_signal_and_rms(df_signal, df_rms, channel_idx=0, n_seconds=3, sampling_rate=200):
    """
    Plota o sinal (geralmente retificado) sobreposto pelo seu cálculo de RMS contínuo.
    """
    limit = int(n_seconds * sampling_rate)
    
    if isinstance(df_signal, np.ndarray):
        df_signal = pd.DataFrame(df_signal)
    if isinstance(df_rms, np.ndarray):
        df_rms = pd.DataFrame(df_rms)
        
    signal_slice = df_signal.iloc[:limit, channel_idx]
    rms_slice = df_rms.iloc[:limit, channel_idx]
    time_axis = np.linspace(0, n_seconds, limit)
    
    plt.figure(figsize=(12, 4))
    
    # Plota o sinal retificado em azul com transparência
    plt.plot(time_axis, signal_slice, color='steelblue', alpha=0.5, label='Rectified Signal')
    
    # Plota o RMS em verde e mais espesso
    plt.plot(time_axis, rms_slice, color='green', linewidth=2, label='RMS Continuous')
    
    plt.title(f'Superposition: Rectified Signal vs RMS (Channel {channel_idx + 1})', fontsize=14)
    plt.xlabel('Time (Seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(loc='upper right')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()