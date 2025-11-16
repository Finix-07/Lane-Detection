# 🐛 Critical Architecture Bug Found and Fixed

## The Root Cause: Sigmoid After Addition

### ❌ Original Code (WRONG):

```python
class BezierRefineHead(nn.Module):
    def forward(self, feat, coarse_pts):
        delta = self.refine(feat_flat)
        refined = coarse_pts + delta.view(...)
        return torch.sigmoid(refined)  # ❌ FATAL BUG!
```

### Why This Causes Mode Collapse:

1. **Coarse predictions** → already in [0, 1] (from sigmoid)
2. **Delta values** → unbounded (can be negative for corrections)
3. **Addition** → `refined = coarse + delta` can go outside [0, 1]
4. **Sigmoid squashing** → brings everything back to [0, 1]

**The Problem:**

- When `refined` goes to large positive values (e.g., 5.0), `sigmoid(5.0) ≈ 0.993`
- When `refined` goes to large negative values (e.g., -5.0), `sigmoid(-5.0) ≈ 0.007`
- **Sigmoid saturates** → gradients become ~0 → model can't learn!

**What the model learns:**

- To avoid gradient death, it learns `delta ≈ 0`
- No refinement happens
- All predictions converge to average values
- Model ignores input image!

### ✅ Fixed Code:

```python
class BezierRefineHead(nn.Module):
    def __init__(self, ...):
        self.refine = nn.Sequential(
            ...,
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, feat, coarse_pts):
        delta = self.refine(feat_flat).view(...)
        delta = delta * 0.2  # Scale to ±0.2 (reasonable refinement)
        refined = coarse_pts + delta
        return torch.clamp(refined, 0.0, 1.0)  # Hard clamp, no sigmoid!
```

**Why this works:**

- `Tanh()` outputs bounded values [-1, 1]
- Scale to ±0.2 (20% of image size - reasonable refinement)
- `clamp()` has gradient=1 in valid range (no saturation!)
- Model can learn meaningful refinements

## Additional Fixes:

### 1. Coarse Head Improvements:

```python
class BezierCoarseHead(nn.Module):
    def __init__(self, ...):
        self.regressor = nn.Sequential(
            ...,
            nn.Dropout(0.1),  # Regularization to prevent overfitting
            ...,
            nn.Sigmoid()     # Sigmoid in Sequential, cleaner
        )
```

### 2. Why Your Training Failed:

**Epochs 1-20:** Model tries to learn, but sigmoid kills gradients
**Epochs 20-30:** Gradients vanish, predictions converge to constants
**Epochs 30-41:** Severe overfitting on training set, but predictions still constant

**The validation loss went up** because:

- Training loss dropped (memorizing data)
- But predictions stayed constant (mode collapse)
- Validation loss increased (overfitting + no actual learning)

## What to Do Now:

### Step 1: Retrain with Fixed Architecture

```bash
python train_fixed.py
```

**Expected behavior:**

- Predictions will be DIFFERENT for each image ✅
- Validation loss should decrease and stabilize
- No more mode collapse

### Step 2: Verify the Fix

After 5-10 epochs, check:

```bash
python check_predictions.py
```

**You should see:**

- Variance > 0.01 (not 0.000000)
- Different predictions per image
- Reasonable control point positions

## Technical Explanation:

### Sigmoid Saturation Problem:

```
sigmoid(x) = 1 / (1 + e^(-x))

When x > 5:  sigmoid(x) ≈ 0.99  → gradient ≈ 0.007
When x < -5: sigmoid(x) ≈ 0.01  → gradient ≈ 0.007
When x = 0:  sigmoid(x) = 0.50  → gradient = 0.25 (max!)
```

**In your case:**

- `refined = coarse + delta` often goes outside [-5, 5]
- Sigmoid gradients become ~0
- Backpropagation fails
- Model learns delta=0 to stay in safe zone

### Clamp vs Sigmoid:

```
clamp(x, 0, 1):
  if x < 0: return 0, gradient = 0
  if x > 1: return 1, gradient = 0
  else: return x, gradient = 1  ← Full gradient in valid range!

sigmoid(x):
  Always has gradient < 0.25
  Gradient → 0 for extreme values
```

**Clamp is better** because:

- Full gradient (1.0) in valid range [0, 1]
- Only kills gradient outside bounds (where it should be clamped anyway)
- No saturation problem

## Summary:

**Root cause:** `sigmoid(coarse + delta)` causes gradient vanishing
**Symptom:** All predictions converge to same values (mode collapse)
**Fix:** Use `clamp(coarse + tanh(delta) * scale, 0, 1)` instead
**Action:** Retrain with fixed architecture

The old checkpoints are **unusable** because they were trained with the buggy architecture. You must retrain from scratch.

---

**Files modified:**

- ✅ `arch.py` - Fixed `BezierRefineHead` and `BezierCoarseHead`

**Next steps:**

1. Run `python train_fixed.py` to retrain
2. After 10 epochs, check predictions with `python check_predictions.py`
3. Expect variance > 0.01 and different predictions per image
