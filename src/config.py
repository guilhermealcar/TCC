"""
Configuration parameters for all NinaPro databases.
Each database has different sampling rates, channels, and gesture counts.
"""

DATABASE_CONFIGS = {
    'DB1': {
        'fs': 100,              # Sampling rate in Hz
        'n_channels': 10,       # Number of EMG electrodes
        'n_gestures': 52,       # Total gesture classes
        'n_repetitions': 10,    # Number of repetitions per subject
        'n_subjects': 27,
        'lowcut': 15.0,         # Bandpass filter low cutoff (Hz)
        'highcut': 45.0,        # Must be less than fs/2 (Nyquist = 50 Hz for DB1)
        'window_ms': 200,       # Window length in milliseconds
        'overlap_pct': 0.5,     # Window overlap (0 to 1)
        'test_reps': [2, 5],    # Repetitions for testing
        'description': '52 gestures, 10 Otto Bock, 100 Hz'
    },
    'DB2': {
        'fs': 2000,
        'n_channels': 12,
        'n_gestures': 49,
        'n_repetitions': 6,
        'n_subjects': 40,
        'lowcut': 20.0,
        'highcut': 500.0,
        'window_ms': 200,
        'overlap_pct': 0.5,
        'test_reps': [2, 5],
        'description': '49 gestures, 12 Delsys, 2000 Hz'
    },
    'DB5': {
        'fs': 200,
        'n_channels': 16,
        'n_gestures': 52,
        'n_repetitions': 6,
        'n_subjects': 10,
        'lowcut': 15.0,
        'highcut': 90.0,       # Must be < fs/2 = 100 Hz
        'window_ms': 200,
        'overlap_pct': 0.5,
        'test_reps': [2, 5],
        'description': '52 gestures, 16 Myo armband, 200 Hz'
    }
}