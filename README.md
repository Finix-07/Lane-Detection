# Lane Detection with Bezier Curves

A modular PyTorch implementation of lane detection using SegFormer backbone and Bezier curve representation.

## 📁 Project Structure

```
Lane-Detection/
├── src/                          # Source code
│   ├── models/                   # Model architectures and losses
│   │   ├── arch.py              # LaneNet architecture
│   │   └── losses.py            # Loss functions
│   ├── data/                    # Data loading and preprocessing
│   │   ├── dataset_loader.py   # PyTorch dataset classes
│   │   └── preprocess_tusimple_bezier.py
│   ├── training/                # Training scripts
│   │   ├── train.py            # Main training script (fixed)
│   │   └── train_legacy.py     # Legacy training script
│   ├── inference/               # Inference and evaluation
│   │   ├── inference.py        # Main inference script (fixed)
│   │   └── inference_legacy.py # Legacy inference script
│   └── utils/                   # Utility functions
│       └── OutputProcess.py    # Bezier curve utilities
├── tests/                       # Unit tests and validation
│   ├── validate_fixes.py       # Architecture validation
│   └── test_loss_params.py     # Loss function tests
├── scripts/                     # Jupyter notebooks and experiments
│   ├── new_model.ipynb
│   └── working.ipynb
├── docs/                        # Documentation
│   ├── ARCHITECTURE_FIXES_APPLIED.md
│   ├── BUGS_FOUND.md
│   ├── BUGS_SUMMARY.md
│   └── ...
├── checkpoints/                 # Model checkpoints
│   ├── production/             # Production-ready models
│   └── experiments/            # Experimental checkpoints
├── outputs/                     # Output artifacts
│   ├── inference/              # Inference results
│   └── visualizations/         # Training curves, etc.
├── tusimple/                    # TuSimple dataset (raw)
├── tusimple_processed/          # Preprocessed data
├── train.py                     # Main training entry point
├── inference.py                 # Main inference entry point
├── preprocess_data.py           # Data preprocessing entry point
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision transformers pillow scipy matplotlib tqdm
```

### 2. Preprocess Dataset

```bash
python preprocess_data.py
```

### 3. Train Model

```bash
python train.py
```

### 4. Run Inference

```bash
python inference.py
```

## 🏗️ Architecture

- **Backbone**: SegFormer MiT-B0 (pretrained)
- **Feature Fusion**: FPN with 128 channels
- **Spatial Modeling**: RESA+ (Recurrent Feature Shift Aggregator)
- **Lane Representation**: Quintic Bezier curves (6 control points)
- **Loss Function**: Multi-task loss (regression + existence + curvature)

## 📊 Model Components

### Multi-Task Heads

1. **Bezier Regression**:
   - Coarse Head: Initial control point predictions
   - Refine Head: Delta refinements for precise localization
2. **Lane Existence**: Binary classification per lane

3. **Strip Proposals**: Auxiliary head for strip-based detection

4. **Segmentation**: Auxiliary pixel-wise lane segmentation

## 🔧 Configuration

Edit `src/training/train.py` CONFIG dict:

```python
CONFIG = {
    "batch_size": 4,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "val_split": 0.1,
    "save_dir": "checkpoints/experiments",
    "save_freq": 5,
    "grad_clip": 1.0,
}
```

## 📈 Training

The training script includes:

- ✅ Fixed loss function (no learnable uncertainty)
- ✅ Gradient clipping for stability
- ✅ Checkpoint saving every N epochs
- ✅ Training curve visualization
- ✅ Validation monitoring

**Expected Results** (after 15-20 epochs):

- Training loss: < 0.10
- Validation loss: < 0.15
- Prediction variance: > 0.01

## 🎯 Inference

```bash
python inference.py
```

Outputs:

- Side-by-side ground truth vs predictions
- Bezier curve visualizations
- Control point annotations
- Saved to `outputs/inference/`

## 🧪 Testing

Run validation tests:

```bash
# Validate architecture fixes
python tests/validate_fixes.py

# Test loss parameters
python tests/test_loss_params.py
```

## 📝 Key Features

### Recent Fixes (All Applied ✅)

1. **Architecture Bugs Fixed**:

   - ✅ Removed duplicate FPN
   - ✅ Removed unused stem/cnn_stage
   - ✅ Fixed sigmoid saturation in refinement head
   - ✅ Safe dtype/device handling
   - ✅ Changed ReLU to inplace=False

2. **Loss Function Fixes**:

   - ✅ Returns tensors (not scalars)
   - ✅ Simple weighted loss (no uncertainty)
   - ✅ Proper gradient flow

3. **Training Improvements**:
   - ✅ Gradient clipping
   - ✅ Better checkpoint management
   - ✅ Training curve plotting

## 📚 Documentation

See `docs/` directory for detailed documentation:

- **ARCHITECTURE_FIXES_APPLIED.md**: Complete list of architecture fixes
- **BUGS_FOUND.md**: Detailed bug analysis
- **BUGS_SUMMARY.md**: Executive summary of fixes

## 🎓 Dataset

Using TuSimple lane detection dataset:

- Training: 3,626 images
- Test: 2,782 images
- Resolution: 1280×720
- Lanes: Up to 5 per image

## 🔬 Model Details

**Total Parameters**: 4.3M

- MiT-B0 Backbone: ~3.7M
- Task Heads: ~600K

**Input**: RGB images (1280×720), normalized
**Output**:

- 6 Bezier control points per lane (max 6 lanes)
- Lane existence logits
- Auxiliary segmentation mask

## 📧 Citation

If you use this code, please cite:

```
@misc{lane-detection-bezier,
  title={Lane Detection with Bezier Curves and SegFormer},
  author={Your Name},
  year={2025}
}
```

## 🎯 TODOs

- [ ] Add TensorBoard logging
- [ ] Add evaluation metrics (F1, accuracy)
- [ ] Add data augmentation
- [ ] Add multi-GPU training support
- [ ] Add model export (ONNX/TorchScript)

## 📄 License

MIT License

---

**Status**: ✅ All bugs fixed, ready for training!
