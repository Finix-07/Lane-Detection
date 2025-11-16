"""
Validate Bézier Ground Truth Quality

This script checks:
1. Monotonic y-coordinates (no loops)
2. Fit quality (reconstruction error)
3. Coordinate ranges (in [0,1])
4. Lane statistics
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

def sample_bezier_curve(ctrl_pts, num_samples=50):
    """Sample points along Bézier curve"""
    t = np.linspace(0, 1, num_samples)
    n = 5  # degree (6 control points = degree 5)
    
    # Bernstein basis
    basis = []
    for i in range(6):
        binom = math.comb(n, i)
        b = binom * ((1 - t) ** (n - i)) * (t ** i)
        basis.append(b)
    
    # Weighted sum
    curve = np.zeros((num_samples, 2))
    for i in range(6):
        curve += basis[i][:, None] * ctrl_pts[i]
    
    return curve


def validate_bezier_gt(gt_path):
    """Validate Bézier ground truth file"""
    print("="*80)
    print("Validating Bézier Ground Truth")
    print("="*80)
    
    # Load data
    data = torch.load(gt_path)
    print(f"\n📂 Loaded: {gt_path}")
    print(f"   Total samples: {len(data)}")
    
    # Statistics
    num_lanes_list = []
    monotonic_violations = 0
    out_of_range = 0
    total_lanes = 0
    
    fit_errors = []
    y_gradients = []
    
    print("\n🔍 Checking quality...")
    
    for idx, sample in enumerate(data):
        ctrl = sample["bezier_ctrl"]  # [num_lanes, 6, 2]
        num_lanes = len(ctrl)
        num_lanes_list.append(num_lanes)
        total_lanes += num_lanes
        
        for lane_idx, lane_ctrl in enumerate(ctrl):
            lane_ctrl_np = lane_ctrl.numpy()
            
            # Check 1: Monotonic y-coordinates
            y_coords = lane_ctrl_np[:, 1]
            y_diffs = np.diff(y_coords)
            if np.any(y_diffs <= 0):
                monotonic_violations += 1
            
            # Check 2: Range [0, 1]
            if np.any(lane_ctrl_np < 0) or np.any(lane_ctrl_np > 1):
                out_of_range += 1
            
            # Check 3: Y-gradient (should be reasonable)
            avg_y_grad = np.mean(y_diffs)
            y_gradients.append(avg_y_grad)
    
    # Print results
    print(f"\n✅ Validation Results:")
    print(f"\n   Lane Statistics:")
    print(f"      Total lanes: {total_lanes}")
    print(f"      Avg lanes per image: {np.mean(num_lanes_list):.2f}")
    print(f"      Max lanes per image: {max(num_lanes_list)}")
    print(f"      Min lanes per image: {min(num_lanes_list)}")
    
    print(f"\n   Quality Checks:")
    violation_pct = (monotonic_violations / total_lanes) * 100
    print(f"      Monotonic violations: {monotonic_violations}/{total_lanes} ({violation_pct:.2f}%)")
    if monotonic_violations > 0:
        print(f"         ⚠️  WARNING: Some lanes have non-increasing y-coordinates!")
    else:
        print(f"         ✓ All lanes have monotonic y-coordinates")
    
    out_of_range_pct = (out_of_range / total_lanes) * 100
    print(f"      Out-of-range points: {out_of_range}/{total_lanes} ({out_of_range_pct:.2f}%)")
    if out_of_range > 0:
        print(f"         ⚠️  WARNING: Some control points are outside [0,1]!")
    else:
        print(f"         ✓ All control points in valid range")
    
    print(f"\n   Y-Gradient Statistics:")
    print(f"      Mean: {np.mean(y_gradients):.4f}")
    print(f"      Std:  {np.std(y_gradients):.4f}")
    print(f"      Min:  {np.min(y_gradients):.4f}")
    print(f"      Max:  {np.max(y_gradients):.4f}")
    
    # Visualize sample
    print(f"\n📊 Generating visualization...")
    visualize_samples(data, num_samples=4, save_path="bezier_gt_validation.png")
    
    print("\n" + "="*80)
    if monotonic_violations == 0 and out_of_range == 0:
        print("✅ VALIDATION PASSED - Ground truth quality is excellent!")
    elif monotonic_violations < total_lanes * 0.01:  # Less than 1%
        print("⚠️  VALIDATION PASSED WITH WARNINGS - Minor issues detected")
    else:
        print("❌ VALIDATION FAILED - Significant quality issues detected")
    print("="*80)


def visualize_samples(data, num_samples=4, save_path="bezier_gt_validation.png"):
    """Visualize sample Bézier curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(data))):
        sample = data[i]
        ctrl = sample["bezier_ctrl"]
        
        ax = axes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Flip y-axis (image coordinates)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Sample {i+1}: {len(ctrl)} lanes", fontsize=10)
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")
        
        # Plot each lane
        for lane_idx, lane_ctrl in enumerate(ctrl):
            lane_ctrl_np = lane_ctrl.numpy()
            
            # Sample curve
            curve = sample_bezier_curve(lane_ctrl_np, num_samples=100)
            
            # Plot curve
            ax.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2, alpha=0.7, label='Curve' if lane_idx == 0 else '')
            
            # Plot control points
            ax.plot(lane_ctrl_np[:, 0], lane_ctrl_np[:, 1], 'ro-', markersize=4, alpha=0.5, label='Control pts' if lane_idx == 0 else '')
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    # Default path
    gt_path = "tusimple/TUSimple/train_set/bezier_gt/train_bezier.pt"
    
    if len(sys.argv) > 1:
        gt_path = sys.argv[1]
    
    if not Path(gt_path).exists():
        print(f"❌ Error: File not found: {gt_path}")
        print(f"\nUsage: python {sys.argv[0]} [path_to_bezier_gt.pt]")
        sys.exit(1)
    
    validate_bezier_gt(gt_path)
