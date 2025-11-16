"""
Configuration settings for Lane Detection project
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_ROOT = PROJECT_ROOT / "tusimple"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "tusimple_processed"

# Checkpoint paths
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PRODUCTION_CHECKPOINT_DIR = CHECKPOINT_DIR / "production"
EXPERIMENT_CHECKPOINT_DIR = CHECKPOINT_DIR / "experiments"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
INFERENCE_OUTPUT_DIR = OUTPUT_DIR / "inference"
VISUALIZATION_OUTPUT_DIR = OUTPUT_DIR / "visualizations"

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 2,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "val_split": 0.1,
    "save_freq": 5,
    "grad_clip": 1.0,
    
    # Loss weights
    "w_reg": 1.0,
    "w_exist": 1.0,
    "w_curv": 0.1,
}

# Model configuration
MODEL_CONFIG = {
    "max_lanes": 6,
    "num_control_points": 6,
    "backbone": "nvidia/mit-b0",
    "fpn_channels": 128,
}

# Data configuration
DATA_CONFIG = {
    "image_height": 720,
    "image_width": 1280,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "mps" if hasattr(os, "mps") else "cpu"
