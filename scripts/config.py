# Configuration file for paths and hyperparameters
import os

# === Base Directories ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'kmms_test', 'kmms_test')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'images')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'masks')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# === Training Hyperparameters ===
PATCH_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 40
LEARNING_RATE = 5e-4

# Add more config options as needed for reproducibility and clarity
