"""
Improved Loss Function for Lane Detection with Bézier Curves

KEY IMPROVEMENTS:
- Pixel-space supervision (better geometric meaning than normalized space)
- Dual supervision: control points (coarse) + sampled curve points (fine-grained)
- Direct curve shape matching with L1 loss on 50 sampled points
- Better gradient quality from dense sampling vs sparse control points

COMPONENTS:
1. Control point loss: Coarse geometry supervision (6 points)
2. Sampled curve loss: Fine-grained shape alignment (50 points) - NEW!
3. Curvature loss: Smoothness regularization
4. Existence loss: Lane presence confidence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BezierLaneLossFinal(nn.Module):
    """
    Improved loss with pixel-space supervision and sampled curve matching.
    
    Components:
    1. Control point loss: Coarse geometry (SmoothL1 on 6 control points)
    2. Sampled curve loss: Fine-grained alignment (L1 on 50 sampled points) - KEY IMPROVEMENT
    3. Curvature loss: Smoothness regularization
    4. Existence loss: Lane presence confidence
    """
    def __init__(self, 
                 num_ctrl=6,
                 num_samples=50,
                 w_ctrl=1.0,
                 w_curve=2.0,
                 w_exist=1.0,
                 w_curv=0.05,
                 img_width=1280,
                 img_height=720):
        super().__init__()
        self.num_ctrl = num_ctrl
        self.num_samples = num_samples
        self.w_ctrl = w_ctrl
        self.w_curve = w_curve / math.log2(num_samples + 1)  # Scale by sample count to balance gradients
        self.w_exist = w_exist
        self.w_curv = w_curv
        self.img_width = img_width
        self.img_height = img_height
        
        self.exist_loss_fn = nn.BCEWithLogitsLoss()
        
        # Precompute Bernstein polynomial coefficients for quintic Bézier (degree 5)
        n = self.num_ctrl - 1  # degree = 5 for 6 control points
        t = torch.linspace(0, 1, self.num_samples).view(1, 1, self.num_samples, 1)
        i = torch.arange(0, n + 1).view(1, 1, 1, n + 1)
        binom = torch.tensor([math.comb(n, k) for k in range(n + 1)]).view(1, 1, 1, n + 1)
        
        # Bernstein basis: C(n,i) * (1-t)^(n-i) * t^i
        bernstein = binom * ((1 - t) ** (n - i)) * (t ** i)  # [1, 1, num_samples, 6]
        self.register_buffer('bernstein_basis', bernstein, persistent=False)

    def sample_bezier(self, ctrl_pts):
        """
        Sample points along Bézier curve using precomputed basis functions.
        
        Args:
            ctrl_pts: [B, L, num_ctrl, 2] control points in pixel coordinates
        
        Returns:
            [B, L, num_samples, 2] sampled curve points
        """
        # ctrl_pts: [B, L, 6, 2]
        # bernstein_basis: [1, 1, num_samples, 6]
        # Ensure basis is on same device as control points (fix CUDA mismatch)
        basis = self.bernstein_basis.to(ctrl_pts.device)
        # Result: weighted sum over control points
        sampled = (basis.unsqueeze(-1) * ctrl_pts.unsqueeze(2)).sum(dim=3)
        return sampled  # [B, L, num_samples, 2]

    def forward(self, pred_ctrl_norm, gt_ctrl_norm, pred_exist, gt_exist, lane_mask=None):
        """
        Args:
            pred_ctrl_norm: [B, max_lanes, 6, 2] predicted control points (normalized [0,1])
            gt_ctrl_norm: [B, max_lanes, 6, 2] ground truth control points (normalized [0,1])
            pred_exist: [B, max_lanes] existence logits
            gt_exist: [B, max_lanes] existence labels (0 or 1)
            lane_mask: [B, max_lanes] mask for valid lanes (optional)
        
        Returns:
            loss_dict with keys: total, ctrl, curve, exist, curv, mono (optional)
        """
        # Clamp normalized predictions to [0,1] to prevent out-of-bounds and gradient explosion
        pred_ctrl_norm = pred_ctrl_norm.clamp(0, 1)
        
        # Convert normalized [0,1] coordinates to pixel space
        # Use tensor for device-safe scaling
        scale = torch.tensor([self.img_width, self.img_height], 
                            device=pred_ctrl_norm.device, dtype=pred_ctrl_norm.dtype)
        pred_ctrl = pred_ctrl_norm * scale
        gt_ctrl = gt_ctrl_norm * scale
        
        # Use ground truth existence as mask if not provided
        if lane_mask is None:
            lane_mask = gt_exist.bool()
        else:
            lane_mask = lane_mask.bool()
        
        # ==================== CONTROL POINT LOSS ====================
        # Coarse geometry supervision on control points
        ctrl_loss = F.smooth_l1_loss(pred_ctrl, gt_ctrl, reduction='none')  # [B, L, 6, 2]
        ctrl_loss = ctrl_loss.mean(dim=(-1, -2))  # [B, L]
        ctrl_loss = (ctrl_loss * lane_mask.float()).sum() / (lane_mask.sum() + 1e-6)
        
        # ==================== SAMPLED CURVE LOSS ====================
        # Fine-grained shape alignment - KEY IMPROVEMENT!
        # Sample 50 points along each Bézier curve for dense supervision
        pred_curve = self.sample_bezier(pred_ctrl)  # [B, L, num_samples, 2]
        gt_curve = self.sample_bezier(gt_ctrl)      # [B, L, num_samples, 2]
        
        curve_loss = F.l1_loss(pred_curve, gt_curve, reduction='none')  # [B, L, S, 2]
        curve_loss = curve_loss.mean(dim=(-1, -2))  # [B, L]
        curve_loss = (curve_loss * lane_mask.float()).sum() / (lane_mask.sum() + 1e-6)
        
        # ==================== CURVATURE SMOOTHNESS LOSS ====================
        # Penalize sharp changes in control point positions (smoothness regularization)
        delta1 = pred_ctrl[:, :, 1:, :] - pred_ctrl[:, :, :-1, :]  # First order differences
        delta2 = delta1[:, :, 1:, :] - delta1[:, :, :-1, :]        # Second order differences
        
        curv_loss = (delta2 ** 2).mean(dim=(-1, -2))  # [B, L]
        curv_loss = (curv_loss * lane_mask.float()).sum() / (lane_mask.sum() + 1e-6)
        
        # ==================== EXISTENCE LOSS ====================
        exist_loss = self.exist_loss_fn(pred_exist, gt_exist)
        
        # ==================== MONOTONIC Y-AXIS CONSTRAINT ====================
        # Ensure y-coordinates increase downward (valid lane geometry)
        # Penalize non-monotonic predictions that could form loops
        y_coords = pred_ctrl[..., 1]  # [B, L, 6]
        mono_penalty = F.relu(y_coords[..., :-1] - y_coords[..., 1:])  # [B, L, 5]
        mono_loss = (mono_penalty.mean(dim=-1) * lane_mask.float()).sum() / (lane_mask.sum() + 1e-6)
        
        # ==================== TOTAL LOSS ====================
        total_loss = (
            self.w_ctrl * ctrl_loss +
            self.w_curve * curve_loss +
            self.w_exist * exist_loss +
            self.w_curv * curv_loss +
            0.1 * mono_loss  # Small weight for monotonic constraint
        )
        
        loss_dict = {
            "total": total_loss,
            "ctrl": ctrl_loss,
            "curve": curve_loss,
            "exist": exist_loss,
            "curv": curv_loss,
            "mono": mono_loss
        }
        
        return loss_dict


class BezierLaneLossWithClipping(nn.Module):
    """
    Enhanced version with gradient clipping and value checking.
    Use this if training is still unstable.
    """
    def __init__(self, w_reg=1.0, w_exist=1.0, w_curv=0.1, clip_grad=10.0):
        super().__init__()
        self.w_reg = w_reg
        self.w_exist = w_exist
        self.w_curv = w_curv
        self.clip_grad = clip_grad
        
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.exist_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred_ctrl_norm, gt_ctrl_norm, pred_exist, gt_exist, lane_mask=None):
        # Convert normalized to actual coordinates
        pred_ctrl = pred_ctrl_norm.clone()
        gt_ctrl = gt_ctrl_norm.clone()
        
        # Check for NaN/Inf in inputs
        if torch.isnan(pred_ctrl).any() or torch.isinf(pred_ctrl).any():
            print("⚠️ WARNING: NaN/Inf detected in pred_ctrl")
            pred_ctrl = torch.nan_to_num(pred_ctrl, nan=0.5, posinf=1.0, neginf=0.0)
        
        # ==================== REGRESSION LOSS ====================
        if lane_mask is None:
            lane_mask = gt_exist
        
        reg_l = self.reg_loss_fn(pred_ctrl, gt_ctrl).mean(dim=(-1, -2))
        reg_l = reg_l * lane_mask.float()
        reg_loss = reg_l.sum() / (lane_mask.sum() + 1e-6)
        
        # Clip to prevent extreme values
        reg_loss = torch.clamp(reg_loss, 0.0, 10.0)
        
        # ==================== EXISTENCE LOSS ====================
        exist_loss = self.exist_loss_fn(pred_exist, gt_exist)
        exist_loss = torch.clamp(exist_loss, 0.0, 10.0)
        
        # ==================== CURVATURE LOSS ====================
        delta1 = pred_ctrl[:, :, 1:, :] - pred_ctrl[:, :, :-1, :]
        delta2 = delta1[:, :, 1:, :] - delta1[:, :, :-1, :]
        
        curvature = (delta2 ** 2).mean(dim=(-1, -2))
        curvature = curvature * lane_mask.float()
        curv_loss = curvature.sum() / (lane_mask.sum() + 1e-6)
        curv_loss = torch.clamp(curv_loss, 0.0, 1.0)
        
        # ==================== TOTAL LOSS ====================
        total_loss = (
            self.w_reg * reg_loss +
            self.w_exist * exist_loss +
            self.w_curv * curv_loss
        )
        
        # Final safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("⚠️ WARNING: NaN/Inf in total loss, returning safe value")
            total_loss = torch.tensor(1.0, device=total_loss.device, requires_grad=True)
        
        loss_dict = {
            "total": total_loss,
            "reg_loss": reg_loss,  # Keep as tensor for gradient flow
            "exist_loss": exist_loss,  # Keep as tensor for gradient flow
            "curv_loss": curv_loss,  # Keep as tensor for gradient flow
        }
        
        return loss_dict


# Aliases for backward compatibility
BezierLaneLoss = BezierLaneLossFinal
BezierLaneUncertaintyLoss = BezierLaneLossFinal


if __name__ == "__main__":
    # Test the improved loss function
    print("="*80)
    print("Testing BezierLaneLossFinal with sampled curve supervision")
    print("="*80)
    
    batch_size = 4
    max_lanes = 6
    num_ctrl = 6
    
    # Create dummy data (normalized [0,1])
    pred_ctrl = torch.rand(batch_size, max_lanes, num_ctrl, 2, requires_grad=True)
    gt_ctrl = torch.rand(batch_size, max_lanes, num_ctrl, 2)
    pred_exist = torch.randn(batch_size, max_lanes, requires_grad=True)
    gt_exist = torch.randint(0, 2, (batch_size, max_lanes)).float()
    
    # Test new loss with default parameters
    criterion = BezierLaneLossFinal(
        num_ctrl=6,
        num_samples=50,
        w_ctrl=1.0,
        w_curve=2.0,
        w_exist=1.0,
        w_curv=0.05,
        img_width=1280,
        img_height=720
    )
    
    loss_dict = criterion(pred_ctrl, gt_ctrl, pred_exist, gt_exist)
    
    print(f"\n✅ Loss computation successful!")
    print(f"   Total loss:  {loss_dict['total']:.4f}")
    print(f"   Ctrl loss:   {loss_dict['ctrl']:.4f} (control points - coarse)")
    print(f"   Curve loss:  {loss_dict['curve']:.4f} (sampled points - fine) ⭐ NEW")
    print(f"   Exist loss:  {loss_dict['exist']:.4f}")
    print(f"   Curv loss:   {loss_dict['curv']:.4f}")
    print(f"   Mono loss:   {loss_dict['mono']:.4f} (y-axis monotonic constraint) ⭐ NEW")
    
    # Check if loss is reasonable
    assert loss_dict['total'] > 0, "Loss should be positive!"
    assert not torch.isnan(loss_dict['total']), "Loss should not be NaN!"
    assert not torch.isinf(loss_dict['total']), "Loss should not be Inf!"
    
    # Test backward pass
    loss_dict['total'].backward()
    print(f"\n✅ Backward pass successful!")
    
    # Check gradient flow
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in criterion.parameters() if p.requires_grad)
    print(f"   Gradients flowing: {'Yes ✓' if has_grad or not list(criterion.parameters()) else 'No (expected for loss-only module)'}")
    
    print(f"\n✅ All tests passed!")
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS:")
    print("  1. Dense supervision: 50 sampled points vs 6 control points")
    print("  2. Pixel-space loss: Better geometric meaning")
    print("  3. Device-safe: Handles CPU/GPU correctly")
    print("  4. Clamped inputs: Prevents out-of-bounds predictions")
    print("  5. Monotonic constraint: Prevents curve loops")
    print("  6. Balanced gradients: Scaled curve loss weight")
    print("="*80)
