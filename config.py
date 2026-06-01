"""
Configuration constants and paths for VoiceGuard.
"""

from pathlib import Path

# Audio preprocessing parameters
SAMPLE_RATE = 16000
CLIP_DURATION = 4  # seconds
N_MFCC = 40
CQT_BINS = 84
HOP_LENGTH = 512

# Decision settings
THRESHOLD = 0.5

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

# Dataset config
KAGGLE_DATASET = "awsaf49/asvpoof-2019-dataset"
LOCAL_DATASET_PATH = "D:\ANN\Projects\VoiceGuard\Dataset"

# Saved models directories and files
SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"
MODEL_PT_PATH = SAVED_MODELS_DIR / "best_model.pt"
MODEL_ONNX_FP32_PATH = SAVED_MODELS_DIR / "model_fp32.onnx"
MODEL_ONNX_FP16_PATH = SAVED_MODELS_DIR / "model_fp16.onnx"
MODEL_METADATA_PATH = SAVED_MODELS_DIR / "model_metadata.json"

# Ensure the saved models directory exists at import time
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
