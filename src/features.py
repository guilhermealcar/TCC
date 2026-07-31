import numpy as np

def get_mav(window_data):
    """
    Calculate the Mean Absolute Value (MAV) of the given window data.
    """
    return np.mean(np.abs(window_data), axis=0)

def get_rms_discrete(window_data):
    """
    Calculate the discrete Root Mean Square (RMS) of the given window data.
    Returns a unique RMS value for each channel in the window.
    """
    return np.sqrt(np.mean(window_data**2, axis=0))

def get_wl(window_data):
    """
    Calculate the Waveform Length (WL) of the given window data.
    WL is the cumulative length of the waveform over the window.
    """
    return np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)

def get_zc(window_data, threshold=1e-5):
    """
    Calculate the Zero Crossing (ZC) count of the given window data.
    ZC counts the number of times the signal crosses zero, considering a threshold to avoid noise.
    """
    crossings = np.diff(np.sign(window_data), axis=0) != 0
    abs_diff = np.abs(np.diff(window_data, axis=0)) >= threshold
    valid_crossings = crossings & abs_diff

    return np.sum(valid_crossings, axis=0)

def get_ssc(window_data, threshold=1e-5):
    """
    Calculate the Slope Sign Change (SSC) count of the given window data.
    SSC counts the number of times the slope of the signal changes sign, considering a threshold to avoid noise.
    """
    diff_signal = np.diff(window_data, axis=0)
    slope_changes = np.diff(np.sign(diff_signal), axis=0) != 0
    abs_diff1 = np.abs(diff_signal[:-1]) >= threshold
    abs_diff2 = np.abs(diff_signal[1:]) >= threshold

    valid_ssc = slope_changes & (abs_diff1 & abs_diff2)

    return np.sum(valid_ssc, axis=0)

def extract_time_domain_features(window_data):
    """
    Extracts time-domain features from the given window data.
    Returns a dictionary containing MAV, RMS, WL, ZC, and SSC for each channel.
    """
    
    mav = get_mav(window_data)
    rms = get_rms_discrete(window_data)
    wl = get_wl(window_data)
    zc = get_zc(window_data)
    ssc = get_ssc(window_data)
    
    return np.hstack([mav, rms, wl, zc, ssc])