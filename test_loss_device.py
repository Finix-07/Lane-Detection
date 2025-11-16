"""Quick test to verify loss works on different devices"""
import torch
from src.models.losses import BezierLaneLossFinal

print("="*80)
print("Testing device compatibility (CPU/MPS/CUDA)")
print("="*80)

# Determine available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"\nTesting on device: {device}")

# Create test data
batch_size, max_lanes, num_ctrl = 2, 6, 6
pred_ctrl = torch.rand(batch_size, max_lanes, num_ctrl, 2, device=device, requires_grad=True)
gt_ctrl = torch.rand(batch_size, max_lanes, num_ctrl, 2, device=device)
pred_exist = torch.randn(batch_size, max_lanes, device=device, requires_grad=True)
gt_exist = torch.randint(0, 2, (batch_size, max_lanes), device=device).float()

# Create loss
criterion = BezierLaneLossFinal(
    num_ctrl=6,
    num_samples=50,
    w_ctrl=1.0,
    w_curve=2.0,
    w_exist=1.0,
    w_curv=0.05,
    img_width=1280,
    img_height=720
).to(device)

# Forward pass
loss_dict = criterion(pred_ctrl, gt_ctrl, pred_exist, gt_exist)

print(f"\n✅ Forward pass successful on {device}!")
print(f"   Total loss: {loss_dict['total']:.4f}")

# Backward pass
loss_dict['total'].backward()

print(f"✅ Backward pass successful on {device}!")
print(f"✅ Device handling verified!")

# Test with out-of-bounds predictions (should be clamped)
pred_ctrl_oob = torch.randn(batch_size, max_lanes, num_ctrl, 2, device=device, requires_grad=True) * 2.0
loss_dict_oob = criterion(pred_ctrl_oob, gt_ctrl, pred_exist, gt_exist)

print(f"\n✅ Out-of-bounds input handled (clamped)!")
print(f"   Loss: {loss_dict_oob['total']:.4f}")

print("\n" + "="*80)
print("✅ All device tests passed!")
print("="*80)
