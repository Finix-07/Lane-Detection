import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import least_squares

# ===============================================================
# 1. Quintic Bézier fitting function (6 control points)
# ===============================================================
def fit_bezier_6pts(points, image_height=720, image_width=1280):
    """
    Fit quintic Bézier curve (6 control points) to lane points.
    
    NOTE: Returns NORMALIZED control points in [0,1] range.
    This matches the model's expected output format.
    """
    # Normalize coordinates to [0,1] range
    x = points[:, 0] / image_width
    y = points[:, 1] / image_height
    t = np.linspace(0, 1, len(points))

    def bezier_curve(ctrl):
        ctrl = ctrl.reshape(6, 2)
        # Quintic Bézier: B(t) = (1-t)⁵P₀ + 5(1-t)⁴tP₁ + 10(1-t)³t²P₂ + 10(1-t)²t³P₃ + 5(1-t)t⁴P₄ + t⁵P₅
        B = (1 - t)[:, None] ** 5 * ctrl[0] \
            + 5 * (1 - t)[:, None] ** 4 * t[:, None] * ctrl[1] \
            + 10 * (1 - t)[:, None] ** 3 * t[:, None] ** 2 * ctrl[2] \
            + 10 * (1 - t)[:, None] ** 2 * t[:, None] ** 3 * ctrl[3] \
            + 5 * (1 - t)[:, None] * t[:, None] ** 4 * ctrl[4] \
            + t[:, None] ** 5 * ctrl[5]
        return B

    def residual(ctrl):
        ctrl_reshaped = ctrl.reshape(6, 2)
        
        # Fit error
        pred = bezier_curve(ctrl)
        fit_error = (pred - np.stack([x, y], axis=1)).ravel()
        
        # Enforce monotonic y-coordinates (downward increasing)
        # Penalize control points where y[i] >= y[i+1]
        y_diffs = np.diff(ctrl_reshaped[:, 1])
        monotonic_penalty = np.maximum(0, -y_diffs) * 10.0  # Strong penalty for violations
        
        # Combine errors
        return np.concatenate([fit_error, monotonic_penalty])

    # Smart initialization: interpolate along lane geometry instead of straight line
    # This follows the actual curve shape for better optimization convergence
    t_init = np.linspace(0, 1, 6)
    t_data = np.linspace(0, 1, len(x))
    init_ctrl = np.stack([
        np.interp(t_init, t_data, x),
        np.interp(t_init, t_data, y)
    ], axis=1).ravel()

    res = least_squares(residual, init_ctrl)
    return torch.tensor(res.x.reshape(6, 2), dtype=torch.float32)


# ===============================================================
# 2. Process a single TuSimple JSON file
# ===============================================================
def process_tusimple_json(json_path, image_height=720, image_width=1280):
    samples = []
    with open(json_path, 'r') as f:
        data = [json.loads(line) for line in f]

    for item in tqdm(data, desc=f"Processing {os.path.basename(json_path)}"):
        img_path = item["raw_file"]
        h_samples = np.array(item["h_samples"])

        lanes_ctrl = []
        for lane_x in item["lanes"]:
            lane_x = np.array(lane_x)
            # Filter valid points (TuSimple uses -2 for missing)
            valid = lane_x > 0
            if valid.sum() < 6:  # Need at least 6 points for quintic fitting
                continue
            
            # Create (x, y) points - TuSimple h_samples are y-coordinates (top=0, bottom=720)
            pts = np.stack([lane_x[valid], h_samples[valid]], axis=1)
            
            # Sort by y-coordinate (top to bottom) to ensure consistent ordering
            pts = pts[pts[:, 1].argsort()]
            
            try:
                ctrl_pts = fit_bezier_6pts(pts, image_height, image_width)
                # Validate: control points should have increasing y (sanity check)
                if not np.all(np.diff(ctrl_pts[:, 1].numpy()) > -0.01):  # Allow small tolerance
                    continue  # Skip if optimization failed to enforce monotonicity
                lanes_ctrl.append(ctrl_pts)
            except Exception as e:
                # Skip lanes where fitting fails
                continue

        if len(lanes_ctrl) > 0:
            samples.append({
                "image_path": img_path,
                "bezier_ctrl": torch.stack(lanes_ctrl)  # [num_lanes, 6, 2]
            })

    return samples


# ===============================================================
# 3. Combine & save results with efficient padding format
# ===============================================================
def main():
    data_root = "tusimple/TUSimple/train_set"
    save_dir = os.path.join(data_root, "bezier_gt")
    os.makedirs(save_dir, exist_ok=True)

    json_files = [
        "label_data_0313.json",
        "label_data_0531.json",
        "label_data_0601.json"
    ]

    all_samples = []
    max_lanes_per_image = 0
    
    for jf in json_files:
        samples = process_tusimple_json(os.path.join(data_root, jf))
        all_samples.extend(samples)
        # Track max lanes for padding
        for sample in samples:
            max_lanes_per_image = max(max_lanes_per_image, len(sample["bezier_ctrl"]))
    
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Max lanes per image: {max_lanes_per_image}")
    
    # Save in original list format (compatible with current dataset loader)
    save_path = os.path.join(save_dir, "train_bezier.pt")
    torch.save(all_samples, save_path)
    print(f"\n✅ Saved {len(all_samples)} samples to {save_path}")
    print(f"\n📝 Format: List of dicts with 'image_path' and 'bezier_ctrl' [num_lanes, 6, 2]")
    print(f"   Control points are NORMALIZED to [0,1] range")
    print(f"   Y-coordinates are monotonically increasing (top→bottom)")

if __name__ == "__main__":
    main()
