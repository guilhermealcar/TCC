import os

# Path Configurations
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

# Dynamic Database Configurations
DATABASE_CONFIGS = {
    'DB1': {'fs': 100.0,  'n_channels': 10, 'window_ms': 200, 'overlap_pct': 0.5, 'description': '10 channels, 100 Hz'},
    'DB2': {'fs': 2000.0, 'n_channels': 12, 'window_ms': 200, 'overlap_pct': 0.5, 'description': '12 channels, 2000 Hz'},
    'DB3': {'fs': 2000.0, 'n_channels': 12, 'window_ms': 200, 'overlap_pct': 0.5, 'description': '12 channels, 2000 Hz'},
    'DB4': {'fs': 2000.0, 'n_channels': 12, 'window_ms': 200, 'overlap_pct': 0.5, 'description': '12 channels, 2000 Hz'},
    'DB5': {'fs': 200.0,  'n_channels': 16, 'window_ms': 200, 'overlap_pct': 0.5, 'description': '16 channels, 200 Hz'},
    'DB6': {'fs': 2000.0, 'n_channels': 14, 'window_ms': 200, 'overlap_pct': 0.5, 'description': '14 channels, 2000 Hz'},
    'DB7': {'fs': 2000.0, 'n_channels': 12, 'window_ms': 200, 'overlap_pct': 0.5, 'description': '12 channels, 2000 Hz'},
}

# Base Filter Thresholds
LOWCUT = 20.0     
HIGHCUT = 450.0   
NOTCH_FREQ = 50.0 
NOTCH_Q = 30.0

# Class Balancing
REST_CLASS = 0
REST_BALANCE_RATIO = 1.5