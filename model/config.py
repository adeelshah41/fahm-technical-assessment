"""
FAHM Mammogram Classification — Configuration
All hyperparameters and paths for training/evaluation.
"""

# ──────────────── Data Paths ────────────────
DATA_ROOT = "./data/jpeg"       # Root folder containing JPEG images (uid subfolders)
CSV_ROOT = "./data/csv"         # Root folder containing CSV metadata files

# ──────────────── Image / Loader ────────────────
IMG_SIZE = 224                  # Input image size (height == width)
BATCH_SIZE = 16                 # Reduced to 16 for EfficientNet-B4 to fit GPU memory
NUM_WORKERS = 4


# ──────────────── Training ────────────────
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ──────────────── Splits & Reproducibility ────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# ──────────────── Model ────────────────
MODEL_SAVE_PATH = "../backend/app/models/model.pt"
