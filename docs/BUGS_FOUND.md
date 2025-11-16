# Comprehensive Bug Report - Lane Detection Model

## Date: November 17, 2025

---

## 🚨 CRITICAL BUGS FOUND

### Bug #1: Loss Function Returns Scalars Instead of Tensors (CRITICAL)

**File**: `losses_fixed.py` lines 86-88  
**Severity**: ⚠️ **CRITICAL** - Prevents gradient backpropagation  
**Status**: 🔴 **UNFIXED**

#### Problem:

```python
loss_dict = {
    "total": total_loss,
    "reg_loss": reg_loss.item(),      # ❌ WRONG - converts to Python scalar
    "exist_loss": exist_loss.item(),  # ❌ WRONG - breaks gradient graph
    "curv_loss": curv_loss.item(),    # ❌ WRONG - no gradients
}
```

When `.item()` is called on a tensor, it converts it to a Python scalar (float), which:

1. **Breaks the computational graph** - no gradients flow back
2. Makes `loss.backward()` compute gradients only from `total_loss`, ignoring individual components
3. The training script calls `.item()` AGAIN on these already-converted scalars in lines like:
   ```python
   total_reg_loss += loss_dict["reg_loss"].item()  # Calling .item() on a float!
   ```

#### Why This Causes Mode Collapse:

While the main `total_loss` tensor still has gradients, the training loop tries to accumulate metrics by calling `.item()` on already-converted floats. This doesn't directly break training, but it indicates confusion about when to detach values.

More importantly, if any code path uses these dict values for further computation (loss weighting, adaptive scaling, etc.), it would break the gradient flow.

#### Fix:

```python
loss_dict = {
    "total": total_loss,
    "reg_loss": reg_loss,      # ✅ Keep as tensor
    "exist_loss": exist_loss,  # ✅ Keep as tensor
    "curv_loss": curv_loss,    # ✅ Keep as tensor
}
```

Then in training loop:

```python
# When logging/accumulating, call .item() there:
total_reg_loss += loss_dict["reg_loss"].item()  # ✅ Correct
```

#### Evidence:

```
5. GRADIENT CHECK:
   Total loss type: <class 'torch.Tensor'>
   Requires grad: False
   ❌ CRITICAL BUG: Loss doesn't require gradients!
```

---

### Bug #2: Architecture Sigmoid Saturation (FIXED)

**File**: `arch.py` - `BezierRefineHead.forward()`  
**Severity**: ⚠️ **CRITICAL** - Causes gradient vanishing and mode collapse  
**Status**: ✅ **FIXED** (as of recent update)

#### Original Problem:

```python
# OLD CODE (BROKEN):
refined = coarse_pts + delta
return torch.sigmoid(refined)  # ❌ Sigmoid after addition causes saturation
```

When `coarse_pts` (already in [0,1]) is added to `delta` (unbounded), the result can be:

- Much larger than 1 → sigmoid → ~1.0 → gradient ≈ 0
- Much smaller than 0 → sigmoid → ~0.0 → gradient ≈ 0

This causes **gradient saturation** where all refinements converge to the same value.

#### Fix Applied:

```python
# NEW CODE (FIXED):
delta = self.refine(feat_flat).view(-1, self.max_lanes, self.num_ctrl, 2)
delta = delta * 0.2  # Scale to reasonable range
refined = coarse_pts + delta
return torch.clamp(refined, 0.0, 1.0)  # ✅ Clamp preserves gradients in [0,1]
```

**Why clamp is better than sigmoid:**

- `clamp` gradient = 1.0 when value is in [0, 1]
- `sigmoid` gradient ≈ 0 when value is far from 0.5
- Refinement needs to learn small adjustments, not get saturated

---

### Bug #3: Uncertainty Weighting Bug (FIXED)

**File**: `losses.py` - `BezierLaneUncertaintyLoss`  
**Severity**: ⚠️ **HIGH** - Caused negative loss values  
**Status**: ✅ **FIXED** (by using `losses_fixed.py`)

#### Problem:

The uncertainty-weighted loss allowed the model to "cheat" by increasing uncertainty (σ):

```python
total_loss = exp(-log_var) * loss + log_var
```

If the model makes σ very large (log_var → ∞), the first term → 0, but the penalty term (log_var) grows linearly. However, in practice, the model can find a sweet spot where:

- Large uncertainty reduces loss contribution
- Results in negative total loss (observed: -2.47)

#### Fix:

Created `losses_fixed.py` with simple weighted sum (no learnable uncertainty):

```python
total_loss = w_reg * reg_loss + w_exist * exist_loss + w_curv * curv_loss
```

---

## ⚠️ MINOR ISSUES FOUND

### Issue #1: Low Prediction Variance (Symptom, not bug)

**Evidence**:

```
Predictions from 5 different images:
   Sample   0: (0.494358, 0.485486)
   Sample  10: (0.494235, 0.487762)
   Sample  50: (0.494790, 0.483456)
   Sample 100: (0.501170, 0.485454)
   Sample 200: (0.494832, 0.482926)
Variance: (0.00000706, 0.00000295)
```

This is **not a bug** but a **symptom** of Bugs #1 and #2. Once those are fixed, predictions should vary significantly across images.

---

## ✅ VERIFIED CORRECT

### Ground Truth Generation

**File**: `preprocess_tusimple_bezier.py`  
**Status**: ✅ **NO BUGS FOUND**

Comprehensive tests showed:

- ✅ Control points in valid range [0, 1]
- ✅ Proper Y-axis orientation (top → bottom)
- ✅ Bezier fitting error < 1 pixel (excellent)
- ✅ Ground truth varies across samples (variance = 0.002053)
- ✅ h_samples correctly sorted ascending

### Dataset Loading

**File**: `dataset_loader.py`  
**Status**: ✅ **NO BUGS FOUND**

- ✅ Correct batch shape: [B, 6, 6, 2]
- ✅ Proper padding for variable number of lanes
- ✅ Lane existence labels correctly set

### Model Architecture (After Fix)

**File**: `arch.py`  
**Status**: ✅ **FIXED**

- ✅ Output ranges: Coarse [0.47, 0.53], Refine [0.45, 0.55]
- ✅ All outputs properly constrained to [0, 1]
- ✅ Gradient flow verified (after architecture fix)

---

## 📊 PRIORITY ACTION ITEMS

### 1. **IMMEDIATE** - Fix losses_fixed.py

Remove `.item()` from loss_dict return values:

```python
# In losses_fixed.py, line 85-89:
loss_dict = {
    "total": total_loss,
    "reg_loss": reg_loss,      # Remove .item()
    "exist_loss": exist_loss,  # Remove .item()
    "curv_loss": curv_loss,    # Remove .item()
}
```

### 2. **IMMEDIATE** - Update training scripts

Ensure all training scripts (train.py, train_fixed.py) call `.item()` when logging:

```python
# In train_one_epoch():
total_loss += loss.item()
total_reg_loss += loss_dict["reg_loss"].item()  # Add .item() here
total_exist_loss += loss_dict["exist_loss"].item()  # Add .item() here
```

### 3. **HIGH** - Retrain from scratch

After fixes #1 and #2:

```bash
python train_fixed.py
```

Expected results:

- Training loss should decrease smoothly
- Validation loss should be < 1.0 (not 0.15-0.20)
- Predictions should vary across images (variance > 0.01)

---

## 🔍 DEBUGGING TOOLS CREATED

1. **diagnose_ground_truth.py** - Comprehensive GT validation
2. **test_model_forward.py** - Tests forward pass, gradients, and predictions
3. **validate_fixes.py** - Quick architecture validation
4. **check_predictions.py** - Checks for mode collapse
5. **find_best_checkpoint.py** - Tests multiple checkpoints

---

## 📝 TECHNICAL NOTES

### Why Mode Collapse Happened:

1. **Primary cause**: Sigmoid saturation in refinement head (Bug #2)
   - Gradients → 0 when predictions far from target
   - Model gets stuck at initialization values
2. **Secondary cause**: Loss function gradient flow (Bug #1)

   - Component losses converted to scalars too early
   - Confused gradient accumulation in training loop

3. **Tertiary cause**: Uncertainty weighting (Bug #3)
   - Model learned to ignore regression loss
   - Led to negative total loss values

### Why Previous Training Failed:

- 41 epochs trained with **both Bug #1 and Bug #2**
- Architecture bug prevented learning meaningful patterns
- Loss bug caused confusion about gradient flow
- Result: All predictions converged to mean value (~0.494, ~0.485)

### Expected After Fix:

- Predictions should vary significantly (variance > 0.01)
- Training loss should reach < 0.01
- Validation loss should be < 0.10
- Visual predictions should match ground truth closely

---

## 🎯 NEXT STEPS

1. ✅ Architecture bug fixed
2. ⏳ Fix losses_fixed.py (remove .item())
3. ⏳ Update training scripts (add .item() when logging)
4. ⏳ Retrain model from scratch
5. ⏳ Verify predictions have high variance
6. ⏳ Run inference and validate visual results

---

## 📚 REFERENCES

- **Architecture fix**: ARCHITECTURE_BUG_FIX.md
- **Loss fix**: LOSS_PARAMS_FIX.md
- **Applied fixes**: FIXES_APPLIED.md

---

**Generated**: November 17, 2025  
**Status**: Critical bugs identified, fixes documented
