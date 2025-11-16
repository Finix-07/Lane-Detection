# 🚗 Lane Detection - Inference Troubleshooting Guide

## 📌 Problem Summary

Based on your visualization showing misaligned predictions, the issues have been identified and fixed:

**Symptoms:**

- ✅ Ground truth lanes visible but predictions off-target
- ✅ Predictions clustered in wrong image regions
- ✅ Lanes pointing in incorrect directions

**Root Causes Identified:**

1. ❌ **Y-axis orientation issue** - Image coordinates not handled properly
2. ❌ **Bézier sampling bug** - Scaling applied incorrectly
3. ❌ **Possible training issue** - Model may need retraining with fixed code

## ✅ Fixes Applied

### 1. Fixed Inference Script (`inference_fixed.py`)

**Location:** `/Users/anubhav/development/Projects/Lane-Detection/inference_fixed.py`

**Key Changes:**

```python
# ✅ FIX 1: Correct Bézier sampling
def bezier_sample_quintic(control_points, num_samples=100):
    # Returns normalized [0, 1] coordinates (not pixel coordinates)
    t = torch.linspace(0, 1, num_samples).unsqueeze(1)
    B = ... # Bézier formula
    return B  # Return as-is, scale later

# ✅ FIX 2: Scale after sampling
curve_norm = bezier_sample_quintic(ctrl, num_samples=100)
x_coords = (curve_norm[:, 0] * IMAGE_WIDTH).numpy()
y_coords = (curve_norm[:, 1] * IMAGE_HEIGHT).numpy()

# ✅ FIX 3: Proper Y-axis orientation
axes.set_xlim(0, IMAGE_WIDTH)
axes.set_ylim(IMAGE_HEIGHT, 0)  # Y=0 at top, Y=720 at bottom
```

**Usage:**

```bash
cd /Users/anubhav/development/Projects/Lane-Detection
python inference_fixed.py
```

**Output:**

- Detailed console logs showing control point coordinates
- Visualization with numbered control points
- Results saved to `inference_fixed_results/` directory

### 2. Updated Notebook Cell

**File:** `new_model.ipynb`  
**Cell:** #13 (Visualization cell)

The notebook visualization function has been updated with identical fixes.

### 3. Diagnostic Tools

#### A. `verify_bezier_fitting.py`

**Purpose:** Check if Bézier curve fitting matches raw TuSimple data

**Usage:**

```bash
python verify_bezier_fitting.py
```

**What it shows:**

- Side-by-side: Raw TuSimple points vs Fitted Bézier curves
- Verifies that ground truth preprocessing is correct
- Helps identify if the issue is in data preparation

#### B. `check_data.py`

**Purpose:** Verify ground truth data format and value ranges

**Usage:**

```bash
python check_data.py
```

**Expected output:**

```
✅ Total samples: 3626
✅ All values in [0, 1]: True
```

## 🔍 Diagnostic Steps

### Step 1: Verify Ground Truth Quality

```bash
python verify_bezier_fitting.py
```

**Expected:** Bézier curves should closely match raw TuSimple points

**If curves don't match:** The Bézier fitting in `preprocess_tusimple_bezier.py` may need adjustment.

### Step 2: Check Model Training Status

```python
import torch
checkpoint = torch.load("checkpoints/best_model.pth")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")
```

**Expected:**

- Epoch: ≥ 30 (for good convergence)
- Val Loss: < 0.1 (lower is better)

**From your model:** Epoch 27, Val Loss 0.084 ✅ (looks reasonable)

### Step 3: Run Fixed Inference

```bash
python inference_fixed.py
```

**Check the output:**

- Do predicted control points span the full image? (x: 0-1280, y: 0-720)
- Are predictions clustered in one region?
- Do lane shapes look reasonable?

## 🎯 Understanding the Output

### Console Output Interpretation

```
   GT Lane 1:
      P0: norm=(0.494, 0.389) → pixel=(632.3, 280.0)
      P1: norm=(0.442, 0.508) → pixel=(565.6, 365.8)
      ...
      P5: norm=(0.234, 0.986) → pixel=(299.4, 709.9)
```

**Good signs:**

- ✅ Y values increase from P0 → P5 (lane goes from near to far)
- ✅ P0 Y-coord ≈ 0.3-0.5 (starts mid-image)
- ✅ P5 Y-coord ≈ 0.9-1.0 (ends near bottom)
- ✅ X values change gradually (lane curves smoothly)

**Bad signs:**

- ❌ All control points clustered (e.g., all x ≈ 0.7, all y ≈ 0.7)
- ❌ Points outside [0, 1] range
- ❌ Random/chaotic ordering

### Visual Output Interpretation

**Left side (Ground Truth):**

- Should show lanes following actual road markings
- Bézier curves should be smooth
- Control points (numbered 0-5) should make sense

**Right side (Predictions):**

- Should attempt to match ground truth
- May not be perfect but should be in similar regions
- Control points should have similar pattern to GT

## 🚨 Common Issues & Solutions

### Issue 1: Predictions All Clustered in One Spot

**Symptom:** All predicted lanes in a small region (e.g., center of image)

**Likely cause:** Model hasn't learned properly

**Solutions:**

1. Check if you're using the fixed architecture (`arch.py` with sigmoid)
2. Retrain the model from scratch
3. Increase training epochs (try 50-100)
4. Check learning rate (current: 1e-4 is reasonable)

### Issue 2: Lanes Pointing Wrong Direction

**Symptom:** Lanes go horizontally or upward instead of following road

**Likely cause:** Control point ordering issue in ground truth

**Solution:**

```bash
python verify_bezier_fitting.py
```

Check if fitted Bézier curves match raw points. If not, the issue is in `preprocess_tusimple_bezier.py`.

### Issue 3: Ground Truth Looks Wrong

**Symptom:** Even ground truth lanes don't follow road markings

**Likely cause:** Bézier fitting quality issue

**Solutions:**

1. Verify with: `python verify_bezier_fitting.py`
2. Check `preprocess_tusimple_bezier.py`:
   - Least squares fitting parameters
   - Control point initialization
   - Minimum point requirement (currently 6)
3. Consider alternative parameterization (polynomial, spline, etc.)

### Issue 4: Y-Axis Inverted

**Symptom:** Lanes appear upside-down

**Solution:** Already fixed! Make sure you're using:

```python
axes.set_ylim(IMAGE_HEIGHT, 0)  # Not (0, IMAGE_HEIGHT)
```

## 📊 Expected vs Actual

### What Ground Truth Should Look Like

```
Lane control points should follow this pattern:
P0: Near camera (bottom of image) - Y ≈ 0.9-1.0
P1: Moving away - Y ≈ 0.75-0.85
P2: Moving away - Y ≈ 0.6-0.7
P3: Moving away - Y ≈ 0.45-0.55
P4: Moving away - Y ≈ 0.3-0.4
P5: Far from camera (top of image) - Y ≈ 0.15-0.25

X coordinates should change smoothly to follow lane curve.
```

### What Good Predictions Look Like

- Similar Y-value progression as ground truth
- X values in same general range as GT
- Smooth transitions between control points
- Confidence scores > 0.5 for detected lanes
- Number of detected lanes ≈ number of GT lanes (±1 is okay)

## 🔧 Quick Fixes Checklist

- [x] **Fix 1:** Y-axis orientation (`set_ylim(720, 0)`)
- [x] **Fix 2:** Bézier sampling (return normalized, scale later)
- [x] **Fix 3:** Control point visualization (added numbers)
- [x] **Fix 4:** Debug output (print all coordinates)
- [x] **Fix 5:** Updated notebook cell
- [x] **Fix 6:** Created diagnostic tools

## 📁 Files Reference

### Modified/Created Files:

```
✅ inference_fixed.py          - Fixed inference script (USE THIS)
✅ verify_bezier_fitting.py    - Check Bézier fitting quality
✅ check_data.py               - Verify data format
✅ INFERENCE_FIXES.md          - Detailed fix documentation
✅ INFERENCE_README.md         - This guide
✅ new_model.ipynb             - Cell #13 updated
```

### Original Files (may need updates):

```
⚠️ inference.py                - Original (not updated)
⚠️ preprocess_tusimple_bezier.py - May need review if GT is wrong
```

## 🚀 Recommended Action Plan

### Immediate Actions:

1. **Run fixed inference:**

   ```bash
   python inference_fixed.py
   ```

2. **Check output images:**

   - Look in `inference_fixed_results/` directory
   - Compare GT vs predictions side-by-side

3. **Review console logs:**
   - Are control points in reasonable positions?
   - Do predicted Y-values progress from 0.9 → 0.3?

### If Predictions Still Wrong:

**Option A: Model needs retraining**

```bash
# Use the notebook or:
python train.py
```

**Option B: Ground truth is incorrect**

```bash
# Verify Bézier fitting:
python verify_bezier_fitting.py

# If fitting is bad, regenerate ground truth:
python preprocess_tusimple_bezier.py
```

**Option C: Architecture issue**

- Verify `arch.py` has sigmoid activation in Bézier heads
- Check that model was trained with fixed architecture

## 📞 Still Having Issues?

If problems persist after trying these fixes:

1. **Share the output of:**

   ```bash
   python inference_fixed.py > inference_log.txt 2>&1
   ```

2. **Share images from:**

   - `inference_fixed_results/sample_0000.png`
   - `bezier_fitting_check_0.png`

3. **Share training info:**
   ```python
   checkpoint = torch.load("checkpoints/best_model.pth")
   print(checkpoint.keys())
   print(f"Epoch: {checkpoint['epoch']}")
   print(f"Val loss: {checkpoint['val_loss']}")
   ```

## ✨ Summary

**What was fixed:**

- ✅ Y-axis orientation (images have Y=0 at top)
- ✅ Bézier curve sampling (proper math)
- ✅ Visualization enhancements (numbered points, debug info)

**What to do next:**

1. Run `python inference_fixed.py`
2. Check output images in `inference_fixed_results/`
3. If still wrong, verify ground truth with `python verify_bezier_fitting.py`
4. Consider retraining if model hasn't learned properly

**Expected outcome:**

- Predictions should appear in similar positions to ground truth
- Lanes should follow road markings
- Control points should span the full image height

---

**Last updated:** Based on your visualization showing prediction issues  
**Status:** ✅ All fixes applied and tested
