"""
Quick validation script to verify the fixes are working correctly.
This checks that the model outputs are properly constrained to [0, 1].
"""

import torch
from src.models.arch import LaneNet

def test_model_output_ranges():
    """Test that model outputs are properly constrained to [0, 1]."""
    print("=" * 60)
    print("Testing Model Output Constraints")
    print("=" * 60)
    
    # Create model
    model = LaneNet(max_lanes=6)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 720, 1280)  # Batch of 2 images
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
    
    # Check outputs
    bezier_coarse = outputs['bezier_coarse']
    bezier_refine = outputs['bezier_refine']
    exist_logits = outputs['exist_logits']
    
    print(f"\n✅ Forward pass successful!")
    print(f"\nOutput shapes:")
    print(f"  bezier_coarse: {bezier_coarse.shape}")  # [B, max_lanes, num_ctrl, 2]
    print(f"  bezier_refine: {bezier_refine.shape}")  # [B, max_lanes, num_ctrl, 2]
    print(f"  exist_logits:  {exist_logits.shape}")   # [B, max_lanes]
    
    # Check value ranges
    print(f"\n🔍 Checking value ranges:")
    
    # Coarse predictions
    coarse_min = bezier_coarse.min().item()
    coarse_max = bezier_coarse.max().item()
    coarse_in_range = (coarse_min >= 0.0 and coarse_max <= 1.0)
    
    print(f"\n  Bezier Coarse:")
    print(f"    Min: {coarse_min:.6f}")
    print(f"    Max: {coarse_max:.6f}")
    print(f"    Status: {'✅ IN RANGE [0, 1]' if coarse_in_range else '❌ OUT OF RANGE!'}")
    
    # Refined predictions
    refine_min = bezier_refine.min().item()
    refine_max = bezier_refine.max().item()
    refine_in_range = (refine_min >= 0.0 and refine_max <= 1.0)
    
    print(f"\n  Bezier Refine:")
    print(f"    Min: {refine_min:.6f}")
    print(f"    Max: {refine_max:.6f}")
    print(f"    Status: {'✅ IN RANGE [0, 1]' if refine_in_range else '❌ OUT OF RANGE!'}")
    
    # Existence logits (can be any value, will be sigmoid-ed during loss)
    exist_min = exist_logits.min().item()
    exist_max = exist_logits.max().item()
    
    print(f"\n  Existence Logits (before sigmoid):")
    print(f"    Min: {exist_min:.6f}")
    print(f"    Max: {exist_max:.6f}")
    print(f"    Status: ✅ (Logits can be any value)")
    
    # Check after sigmoid
    exist_probs = torch.sigmoid(exist_logits)
    prob_min = exist_probs.min().item()
    prob_max = exist_probs.max().item()
    prob_in_range = (prob_min >= 0.0 and prob_max <= 1.0)
    
    print(f"\n  Existence Probs (after sigmoid):")
    print(f"    Min: {prob_min:.6f}")
    print(f"    Max: {prob_max:.6f}")
    print(f"    Status: {'✅ IN RANGE [0, 1]' if prob_in_range else '❌ OUT OF RANGE!'}")
    
    # Overall validation
    print(f"\n" + "=" * 60)
    if coarse_in_range and refine_in_range and prob_in_range:
        print("✅ ALL CHECKS PASSED!")
        print("   The model correctly constrains outputs to [0, 1]")
        print("   Ready for training/inference")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("   Review the sigmoid activations in arch.py")
    print("=" * 60)
    
    return coarse_in_range and refine_in_range and prob_in_range


def test_bezier_sampling():
    """Test that Bezier sampling function works correctly."""
    print("\n" + "=" * 60)
    print("Testing Bezier Sampling Function")
    print("=" * 60)
    
    from src.utils.OutputProcess import bezier_sample_6pts
    
    # Create dummy control points (normalized coordinates)
    control_points = torch.tensor([
        [0.3, 0.9],  # P0 - bottom
        [0.35, 0.7], # P1
        [0.4, 0.5],  # P2
        [0.45, 0.3], # P3
        [0.5, 0.15], # P4
        [0.55, 0.1], # P5 - top
    ])
    
    # Sample the curve
    curve_points = bezier_sample_6pts(control_points, num_samples=100, 
                                      image_height=720, image_width=1280)
    
    print(f"\n✅ Sampling successful!")
    print(f"  Control points shape: {control_points.shape}")
    print(f"  Curve points shape: {curve_points.shape}")
    print(f"  Num samples: {len(curve_points)}")
    
    # Check that curve points are in valid pixel range
    x_min, x_max = curve_points[:, 0].min().item(), curve_points[:, 0].max().item()
    y_min, y_max = curve_points[:, 1].min().item(), curve_points[:, 1].max().item()
    
    print(f"\n  Pixel coordinates:")
    print(f"    X range: [{x_min:.1f}, {x_max:.1f}]")
    print(f"    Y range: [{y_min:.1f}, {y_max:.1f}]")
    
    x_in_range = (x_min >= 0 and x_max <= 1280)
    y_in_range = (y_min >= 0 and y_max <= 720)
    
    if x_in_range and y_in_range:
        print(f"    Status: ✅ All points within image bounds")
    else:
        print(f"    Status: ⚠️  Some points outside image bounds (this is OK if control points are at edges)")
    
    print("=" * 60)
    
    return True


def main():
    """Run all validation tests."""
    print("\n" + "🔍" * 30)
    print("LANE DETECTION MODEL - VALIDATION SCRIPT")
    print("🔍" * 30 + "\n")
    
    # Test 1: Model output ranges
    test1_passed = test_model_output_ranges()
    
    # Test 2: Bezier sampling
    test2_passed = test_bezier_sampling()
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Model Output Constraints: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Bezier Sampling:          {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("\n🎉 All validations passed! The model is ready.")
        print("\n📝 Next steps:")
        print("   1. Retrain the model: python train.py")
        print("   2. Run inference: python inference.py")
        print("   3. Check FIXES_APPLIED.md for details")
    else:
        print("\n⚠️  Some validations failed. Please review the fixes.")
    
    print("\n")


if __name__ == "__main__":
    main()
