# src/config.py
import os
import torch

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data/raw')
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'data/preprocessed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NinaPro DB1 Movement Lexicon (0-52)
MOVEMENT_LABELS = {
    0: "Rest",
    1: "Index flexion", 2: "Index extension", 3: "Middle flexion", 4: "Middle extension",
    5: "Ring flexion", 6: "Ring extension", 7: "Little finger flexion", 8: "Little finger extension",
    9: "Thumb adduction", 10: "Thumb abduction", 11: "Thumb flexion", 12: "Thumb extension",
    13: "Thumb up", 14: "Extension of index and middle, flexion of others",
    15: "Flexion of ring and little finger, extension of others", 16: "Thumb opposing base of little finger",
    17: "Abduction of all fingers", 18: "Fingers flexed together in fist", 19: "Pointing index",
    20: "Adduction of extended fingers", 21: "Wrist supination (axis: middle finger)",
    22: "Wrist pronation (axis: middle finger)", 23: "Wrist supination (axis: little finger)",
    24: "Wrist pronation (axis: little finger)", 25: "Wrist flexion", 26: "Wrist extension",
    27: "Wrist radial deviation", 28: "Wrist ulnar deviation", 29: "Wrist extension with closed hand",
    30: "Large diameter grasp", 31: "Small diameter grasp (power grip)", 32: "Fixed hook grasp",
    33: "Index finger extension grasp", 34: "Medium wrap", 35: "Ring grasp", 36: "Prismatic four fingers grasp",
    37: "Stick grasp", 38: "Writing tripod grasp", 39: "Power sphere grasp", 40: "Three finger sphere grasp",
    41: "Precision sphere grasp", 42: "Tripod grasp", 43: "Prismatic pinch grasp", 44: "Tip pinch grasp",
    45: "Quadpod grasp", 46: "Lateral grasp", 47: "Parallel extension grasp", 48: "Extension type grasp",
    49: "Power disk grasp", 50: "Open a bottle with a tripod grasp", 51: "Turn a screw", 52: "Cut something"
}