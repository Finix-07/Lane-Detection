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
    # Normalize coordinates
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
        pred = bezier_curve(ctrl)
        return (pred - np.stack([x, y], axis=1)).ravel()

    # Initialize control points evenly spaced
    init_ctrl = np.stack([
        np.linspace(x[0], x[-1], 6),
        np.linspace(y[0], y[-1], 6)
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
            valid = lane_x > 0
            if valid.sum() < 6:  # Need at least 6 points for quintic fitting
                continue
            pts = np.stack([lane_x[valid], h_samples[valid]], axis=1)
            ctrl_pts = fit_bezier_6pts(pts, image_height, image_width)
            lanes_ctrl.append(ctrl_pts)

        if len(lanes_ctrl) > 0:
            samples.append({
                "image_path": img_path,
                "bezier_ctrl": torch.stack(lanes_ctrl)  # [num_lanes, 6, 2]
            })

    return samples


# ===============================================================
# 3. Combine & save results
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
    for jf in json_files:
        samples = process_tusimple_json(os.path.join(data_root, jf))
        all_samples.extend(samples)

    save_path = os.path.join(save_dir, "train_bezier.pt")
    torch.save(all_samples, save_path)
    print(f"\n✅ Saved {len(all_samples)} samples to {save_path}")

if __name__ == "__main__":
    main()
