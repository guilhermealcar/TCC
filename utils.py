import io
import numpy as np
import pandas as pd
import scipy.io as sio
import zipfile
from scipy.signal import butter, filtfilt

def load_emg_data(zip_path, emg_mat_index):
    '''
    Loads a .mat file from a .zip archive and returns it as a pandas DataFrame.
    '''
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        mat_files = [f for f in zip_ref.namelist() if f.endswith('.mat')]
        with zip_ref.open(mat_files[emg_mat_index]) as mat_file:
            mat_data = sio.loadmat(io.BytesIO(mat_file.read()))
            # Transform into DataFrame
            df_emg = pd.DataFrame(mat_data['emg'])
            stimulus = mat_data['stimulus']
            restimulus = mat_data['restimulus']
            return df_emg, stimulus, restimulus
        
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    '''
    Applies a Butterworth bandpass filter to the data.
    '''
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data