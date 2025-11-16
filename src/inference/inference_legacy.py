import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from src.models.arch import LaneNet

# =====================================================
# Configuration
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# =====================================================
# Quintic Bézier Sampling (6 control points)
# =====================================================
def bezier_sample_quintic(control_points, num_samples=100):
    """
    Sample a quintic Bézier curve with 6 control points.
    Formula: B(t) = (1-t)⁵P₀ + 5(1-t)⁴tP₁ + 10(1-t)³t²P₂ + 10(1-t)²t³P₃ + 5(1-t)t⁴P₄ + t⁵P₅
    
    Args:
        control_points: Tensor [6, 2] in normalized coordinates [0,1]
        num_samples: Number of points to sample along the curve
    
    Returns:
        Tensor [num_samples, 2] (x, y) in pixel coordinates
    """
    t = torch.linspace(0, 1, num_samples).unsqueeze(1).to(control_points.device)
    
    # Quintic Bézier coefficients
    B = (1 - t) ** 5 * control_points[0] \
        + 5 * (1 - t) ** 4 * t * control_points[1] \
        + 10 * (1 - t) ** 3 * t ** 2 * control_points[2] \
        + 10 * (1 - t) ** 2 * t ** 3 * control_points[3] \
        + 5 * (1 - t) * t ** 4 * control_points[4] \
        + t ** 5 * control_points[5]
    
    # Scale to pixel coordinates
    B[:, 0] = B[:, 0] * IMAGE_WIDTH
    B[:, 1] = B[:, 1] * IMAGE_HEIGHT
    
    return B

# =====================================================
# Load Model
# =====================================================
def load_model(checkpoint_path, device=DEVICE):
    """Load trained model from checkpoint."""
    model = LaneNet(max_lanes=6).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")
    return model

# =====================================================
# Image Preprocessing
# =====================================================
def preprocess_image(image_path):
    """Load and preprocess image for model inference."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    
    # Transform (same as training)
    transform = T.Compose([
        T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

# =====================================================
# Inference
# =====================================================
@torch.no_grad()
def inference(model, image_tensor, confidence_threshold=0.5):
    """
    Run inference on a single image.
    
    Returns:
        predictions: dict with keys:
            - bezier_curves: list of [6, 2] tensors (normalized coordinates)
            - confidence: list of confidence scores
            - exist_probs: tensor of existence probabilities
    """
    model.eval()
    image_tensor = image_tensor.to(DEVICE)
    
    outputs = model(image_tensor)
    
    # Extract predictions
    bezier_refine = outputs['bezier_refine'][0].cpu()  # [max_lanes, 6, 2]
    exist_logits = outputs['exist_logits'][0].cpu()    # [max_lanes]
    exist_probs = torch.sigmoid(exist_logits)          # Convert to probabilities
    
    # Filter lanes by confidence
    predictions = {
        'bezier_curves': [],
        'confidence': [],
        'exist_probs': exist_probs
    }
    
    for i in range(len(bezier_refine)):
        if exist_probs[i] >= confidence_threshold:
            predictions['bezier_curves'].append(bezier_refine[i])  # [6, 2]
            predictions['confidence'].append(exist_probs[i].item())
    
    return predictions

# =====================================================
# Visualization
# =====================================================
def visualize_predictions(image, predictions, title="Lane Detection", save_path=None):
    """
    Visualize lane predictions on the original image.
    
    Args:
        image: PIL Image or numpy array
        predictions: dict from inference()
        title: plot title
        save_path: path to save the figure
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(title)
    
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    
    for idx, (ctrl_pts, conf) in enumerate(zip(predictions['bezier_curves'], predictions['confidence'])):
        # Sample the Bézier curve
        curve_points = bezier_sample_quintic(ctrl_pts, num_samples=100)
        
        x_coords = curve_points[:, 0].numpy()
        y_coords = curve_points[:, 1].numpy()
        
        # Filter out points outside image bounds
        valid_mask = (x_coords >= 0) & (x_coords < IMAGE_WIDTH) & \
                     (y_coords >= 0) & (y_coords < IMAGE_HEIGHT)
        
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        
        # Plot the lane
        color = colors[idx % len(colors)]
        plt.plot(x_coords, y_coords, color=color, linewidth=3, 
                label=f'Lane {idx+1} ({conf:.2f})', alpha=0.8)
        
        # Plot control points
        ctrl_pixel = ctrl_pts.clone()
        ctrl_pixel[:, 0] *= IMAGE_WIDTH
        ctrl_pixel[:, 1] *= IMAGE_HEIGHT
        plt.scatter(ctrl_pixel[:, 0], ctrl_pixel[:, 1], 
                   color=color, s=50, marker='o', alpha=0.6, zorder=5)
    
    plt.legend(loc='upper right')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    
    plt.show()

# =====================================================
# Compare Ground Truth vs Predictions
# =====================================================
def visualize_comparison(image, predictions, ground_truth=None, save_path=None):
    """
    Side-by-side comparison of ground truth and predictions.
    
    Args:
        image: PIL Image
        predictions: dict from inference()
        ground_truth: dict with 'bezier_ctrl' and 'lane_exist' (optional)
        save_path: path to save the figure
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes = [axes]
    
    # Plot predictions
    axes[0].imshow(image_np)
    axes[0].set_title("Predictions", fontsize=16, fontweight='bold')
    
    colors_pred = ['lime', 'cyan', 'yellow', 'magenta', 'orange', 'pink']
    for idx, (ctrl_pts, conf) in enumerate(zip(predictions['bezier_curves'], predictions['confidence'])):
        curve_points = bezier_sample_quintic(ctrl_pts, num_samples=100)
        x_coords = curve_points[:, 0].numpy()
        y_coords = curve_points[:, 1].numpy()
        
        valid_mask = (x_coords >= 0) & (x_coords < IMAGE_WIDTH) & \
                     (y_coords >= 0) & (y_coords < IMAGE_HEIGHT)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        
        color = colors_pred[idx % len(colors_pred)]
        axes[0].plot(x_coords, y_coords, color=color, linewidth=3, 
                    label=f'Lane {idx+1} ({conf:.2f})', alpha=0.9)
    
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].axis('off')
    
    # Plot ground truth if provided
    if ground_truth is not None:
        axes[1].imshow(image_np)
        axes[1].set_title("Ground Truth", fontsize=16, fontweight='bold')
        
        colors_gt = ['red', 'orange', 'purple', 'brown', 'blue', 'green']
        gt_ctrl = ground_truth['bezier_ctrl']  # [max_lanes, 6, 2]
        gt_exist = ground_truth['lane_exist']  # [max_lanes]
        
        for idx in range(len(gt_ctrl)):
            if gt_exist[idx] == 0:
                continue
            
            curve_points = bezier_sample_quintic(gt_ctrl[idx], num_samples=100)
            x_coords = curve_points[:, 0].numpy()
            y_coords = curve_points[:, 1].numpy()
            
            valid_mask = (x_coords >= 0) & (x_coords < IMAGE_WIDTH) & \
                         (y_coords >= 0) & (y_coords < IMAGE_HEIGHT)
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            
            color = colors_gt[idx % len(colors_gt)]
            axes[1].plot(x_coords, y_coords, color=color, linewidth=3, 
                        label=f'Lane {idx+1}', alpha=0.9)
        
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved comparison to {save_path}")
    
    plt.show()

# =====================================================
# Main Function
# =====================================================
def main():
    # Configuration
    checkpoint_path = "checkpoints/best_model.pth"
    test_image_path = "tusimple/TUSimple/test_set/clips/0530/1492626047222176976_0/20.jpg"
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Load and preprocess image
    image_tensor, original_image = preprocess_image(test_image_path)
    
    # Run inference
    predictions = inference(model, image_tensor, confidence_threshold=0.5)
    
    print(f"\n🎯 Inference Results:")
    print(f"   Detected lanes: {len(predictions['bezier_curves'])}")
    print(f"   Confidence scores: {predictions['confidence']}")
    
    # Visualize
    save_path = os.path.join(output_dir, "prediction.png")
    visualize_predictions(original_image, predictions, 
                         title="Lane Detection Results", 
                         save_path=save_path)

if __name__ == "__main__":
    main()
