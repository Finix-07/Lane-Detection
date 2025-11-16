"""
Fixed Lane Detection Inference Script

Key fixes:
1. Proper Y-axis handling (Y=0 at top, Y=720 at bottom)
2. Correct Bézier curve sampling with proper orientation
3. Better visualization with control point ordering
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from src.models.arch import LaneNet
from src.data.dataset_loader import TuSimpleBezierDataset

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

def bezier_sample_quintic(control_points, num_samples=100):
    """
    Sample a quintic Bézier curve with 6 control points.
    Formula: B(t) = (1-t)⁵P₀ + 5(1-t)⁴tP₁ + 10(1-t)³t²P₂ + 10(1-t)²t³P₃ + 5(1-t)t⁴P₄ + t⁵P₅
    
    Args:
        control_points: Tensor [6, 2] in normalized coordinates [0, 1]
        num_samples: Number of points to sample along the curve
    
    Returns:
        Tensor [num_samples, 2] (x, y) coordinates
    """
    t = torch.linspace(0, 1, num_samples).unsqueeze(1).to(control_points.device)
    
    # Quintic Bézier formula
    B = (1 - t) ** 5 * control_points[0] \
        + 5 * (1 - t) ** 4 * t * control_points[1] \
        + 10 * (1 - t) ** 3 * t ** 2 * control_points[2] \
        + 10 * (1 - t) ** 2 * t ** 3 * control_points[3] \
        + 5 * (1 - t) * t ** 4 * control_points[4] \
        + t ** 5 * control_points[5]
    
    return B


def denormalize_image(img_tensor):
    """Convert normalized image tensor back to displayable RGB."""
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img_np, 0, 1)


def load_model(checkpoint_path, device=DEVICE):
    """Load trained model from checkpoint."""
    print(f"📦 Loading model from {checkpoint_path}")
    model = LaneNet(max_lanes=6).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('val_loss', 'N/A')
        print(f"   ✅ Model loaded (Epoch: {epoch}, Val Loss: {val_loss})")
    else:
        model.load_state_dict(checkpoint)
        print(f"   ✅ Model state dict loaded")
    
    model.eval()
    return model


def visualize_comparison(model, dataset, idx=0, save_path=None):
    """
    Create side-by-side comparison of ground truth and predictions.
    
    Args:
        model: Trained LaneNet model
        dataset: TuSimpleBezierDataset instance
        idx: Sample index to visualize
        save_path: Path to save the figure (optional)
    """
    print(f"\n{'='*80}")
    print(f"VISUALIZING SAMPLE {idx}")
    print(f"{'='*80}")
    
    # Get data
    img_tensor, target = dataset[idx]
    img_np = denormalize_image(img_tensor)
    
    # Ground truth
    gt_ctrl = target["bezier_ctrl"]  # [max_lanes, 6, 2]
    gt_exist = target["lane_exist"]  # [max_lanes]
    num_gt_lanes = int(target["num_lanes"])
    
    print(f"\n📋 Ground Truth: {num_gt_lanes} lanes")
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        img_batch = img_tensor.unsqueeze(0).to(DEVICE)
        outputs = model(img_batch)
        
        pred_ctrl = outputs["bezier_refine"][0].cpu()  # [max_lanes, 6, 2]
        pred_exist_logits = outputs["exist_logits"][0].cpu()
        pred_exist_probs = torch.sigmoid(pred_exist_logits)
    
    # Count predicted lanes
    num_pred_lanes = (pred_exist_probs >= 0.5).sum().item()
    print(f"🎯 Predictions: {num_pred_lanes} lanes")
    print(f"   Existence probabilities: {[f'{p:.3f}' for p in pred_exist_probs.tolist()]}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # ==================== GROUND TRUTH ====================
    axes[0].imshow(img_np)
    axes[0].set_title(f"Ground Truth ({num_gt_lanes} lanes)", fontsize=16, fontweight='bold')
    
    colors_gt = ['red', 'orange', 'purple', 'brown', 'blue', 'pink']
    
    for lane_idx in range(num_gt_lanes):
        if gt_exist[lane_idx] == 0:
            continue
        
        # Get control points (normalized [0, 1])
        ctrl_norm = gt_ctrl[lane_idx]  # [6, 2]
        
        # Debug: Print control points
        print(f"\n   GT Lane {lane_idx + 1}:")
        for i in range(6):
            x_norm, y_norm = ctrl_norm[i]
            x_pix = x_norm * IMAGE_WIDTH
            y_pix = y_norm * IMAGE_HEIGHT
            print(f"      P{i}: norm=({x_norm:.3f}, {y_norm:.3f}) → pixel=({x_pix:.1f}, {y_pix:.1f})")
        
        # Sample the Bézier curve in normalized space
        curve_norm = bezier_sample_quintic(ctrl_norm, num_samples=100)
        
        # Convert to pixel coordinates
        x_coords = (curve_norm[:, 0] * IMAGE_WIDTH).numpy()
        y_coords = (curve_norm[:, 1] * IMAGE_HEIGHT).numpy()
        
        # Filter valid points (within image bounds)
        valid_mask = (x_coords >= 0) & (x_coords < IMAGE_WIDTH) & \
                     (y_coords >= 0) & (y_coords < IMAGE_HEIGHT)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        
        if len(x_coords) == 0:
            print(f"      ⚠️ No valid points after filtering!")
            continue
        
        # Plot lane curve
        color = colors_gt[lane_idx % len(colors_gt)]
        axes[0].plot(x_coords, y_coords, color=color, linewidth=3, 
                    label=f'Lane {lane_idx+1}', alpha=0.9, zorder=2)
        
        # Plot control points with numbers
        ctrl_pixel = ctrl_norm.clone()
        ctrl_pixel[:, 0] *= IMAGE_WIDTH
        ctrl_pixel[:, 1] *= IMAGE_HEIGHT
        
        axes[0].scatter(ctrl_pixel[:, 0], ctrl_pixel[:, 1], 
                       color=color, s=80, marker='o', alpha=0.7, 
                       edgecolors='white', linewidths=2, zorder=3)
        
        # Add point numbers
        for i, (x, y) in enumerate(ctrl_pixel):
            axes[0].text(x, y, str(i), color='white', fontsize=8, 
                        ha='center', va='center', fontweight='bold', zorder=4)
    
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].axis('off')
    axes[0].set_xlim(0, IMAGE_WIDTH)
    axes[0].set_ylim(IMAGE_HEIGHT, 0)  # Flip Y-axis (0 at top)
    
    # ==================== PREDICTIONS ====================
    axes[1].imshow(img_np)
    axes[1].set_title(f"Predictions ({num_pred_lanes} lanes)", fontsize=16, fontweight='bold')
    
    colors_pred = ['lime', 'cyan', 'yellow', 'magenta', 'orange', 'white']
    
    for lane_idx in range(len(pred_ctrl)):
        if pred_exist_probs[lane_idx] < 0.5:
            continue
        
        # Get control points (normalized [0, 1])
        ctrl_norm = pred_ctrl[lane_idx]  # [6, 2]
        
        # Debug: Print control points
        print(f"\n   Pred Lane {lane_idx + 1} (conf={pred_exist_probs[lane_idx]:.3f}):")
        for i in range(6):
            x_norm, y_norm = ctrl_norm[i]
            x_pix = x_norm * IMAGE_WIDTH
            y_pix = y_norm * IMAGE_HEIGHT
            print(f"      P{i}: norm=({x_norm:.3f}, {y_norm:.3f}) → pixel=({x_pix:.1f}, {y_pix:.1f})")
        
        # Check if predictions are in valid range
        if ctrl_norm.min() < -0.1 or ctrl_norm.max() > 1.1:
            print(f"      ⚠️ WARNING: Control points outside expected [0, 1] range!")
            print(f"         Min: {ctrl_norm.min():.3f}, Max: {ctrl_norm.max():.3f}")
        
        # Sample the Bézier curve in normalized space
        curve_norm = bezier_sample_quintic(ctrl_norm, num_samples=100)
        
        # Convert to pixel coordinates
        x_coords = (curve_norm[:, 0] * IMAGE_WIDTH).numpy()
        y_coords = (curve_norm[:, 1] * IMAGE_HEIGHT).numpy()
        
        # Filter valid points (within image bounds)
        valid_mask = (x_coords >= 0) & (x_coords < IMAGE_WIDTH) & \
                     (y_coords >= 0) & (y_coords < IMAGE_HEIGHT)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        
        if len(x_coords) == 0:
            print(f"      ⚠️ No valid points after filtering!")
            continue
        
        # Plot lane curve
        color = colors_pred[lane_idx % len(colors_pred)]
        axes[1].plot(x_coords, y_coords, color=color, linewidth=3, 
                    label=f'Lane {lane_idx+1} ({pred_exist_probs[lane_idx]:.2f})', 
                    alpha=0.9, zorder=2)
        
        # Plot control points with numbers
        ctrl_pixel = ctrl_norm.clone()
        ctrl_pixel[:, 0] *= IMAGE_WIDTH
        ctrl_pixel[:, 1] *= IMAGE_HEIGHT
        
        axes[1].scatter(ctrl_pixel[:, 0], ctrl_pixel[:, 1], 
                       color=color, s=80, marker='o', alpha=0.7,
                       edgecolors='white', linewidths=2, zorder=3)
        
        # Add point numbers
        for i, (x, y) in enumerate(ctrl_pixel):
            axes[1].text(x, y, str(i), color='white', fontsize=8, 
                        ha='center', va='center', fontweight='bold', zorder=4)
    
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].axis('off')
    axes[1].set_xlim(0, IMAGE_WIDTH)
    axes[1].set_ylim(IMAGE_HEIGHT, 0)  # Flip Y-axis (0 at top)
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = f"inference_fixed_sample_{idx}.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {save_path}")
    plt.close()


def main():
    """Main function to run inference on multiple samples."""
    print(f"🚀 Fixed Lane Detection Inference")
    print(f"   Device: {DEVICE}")
    print(f"   Image size: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    
    # Load model
    checkpoint_path = "checkpoints/production/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("   Looking for alternative checkpoints...")
        
        if os.path.exists("checkpoints"):
            checkpoints = [f for f in os.listdir("checkpoints") if f.endswith('.pth')]
            if checkpoints:
                checkpoint_path = os.path.join("checkpoints", checkpoints[0])
                print(f"   Using: {checkpoint_path}")
            else:
                print("   No checkpoints found!")
                return
        else:
            print("   Checkpoints directory not found!")
            return
    
    model = load_model(checkpoint_path)
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    dataset = TuSimpleBezierDataset(split="train")
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Create output directory
    output_dir = "outputs/inference"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Visualize multiple samples
    sample_indices = [0, 50, 100, 200, 500, 1000, 1500, 2000]
    
    for idx in sample_indices:
        if idx >= len(dataset):
            print(f"⚠️ Sample {idx} out of range (max: {len(dataset)-1})")
            continue
        
        save_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
        visualize_comparison(model, dataset, idx=idx, save_path=save_path)
    
    print(f"\n{'='*80}")
    print(f"✅ Inference completed!")
    print(f"   Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
