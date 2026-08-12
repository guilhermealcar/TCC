import os
import zipfile
import h5py
import scipy.io
import numpy as np

def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extracts a ZIP file to a specified directory.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    
    print(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_ninapro_mat(file_path: str) -> dict[str, np.ndarray]:
    """
    Robustly loads a NinaPro .mat file handling both HDF5 (v7.3) and legacy formats.
    Extracts the EMG matrix, stimulus (labels), and repetitions.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
        
    try:
        # Attempt HDF5 loading (Standard for NinaPro DB1/DB2)
        with h5py.File(file_path, 'r') as mat:
            emg = np.array(mat['emg']).T.astype(np.float32)
            stim_key = 'restimulus' if 'restimulus' in mat else 'stimulus'
            rep_key = 'rerepetition' if 'rerepetition' in mat else 'repetition'
            
            labels = np.array(mat[stim_key]).T.flatten().astype(np.int16)
            reps = np.array(mat[rep_key]).T.flatten().astype(np.int16)
            
    except OSError:
        # Fallback to scipy.io for older MATLAB formats
        mat = scipy.io.loadmat(file_path)
        emg = mat['emg'].astype(np.float32)
        stim_key = 'restimulus' if 'restimulus' in mat else 'stimulus'
        rep_key = 'rerepetition' if 'rerepetition' in mat else 'repetition'
        
        labels = mat[stim_key].flatten().astype(np.int16)
        reps = mat[rep_key].flatten().astype(np.int16)
        
    return {
        'emg': emg,
        'labels': labels,
        'reps': reps
    }