import torch
from scipy.optimize import least_squares
import json
import numpy as np


def bezier_sample_6pts(control_points, num_samples=56, image_height=720, image_width=1280):
    """
    Sample a quintic Bézier curve with 6 control points.
    Formula: B(t) = (1-t)⁵P₀ + 5(1-t)⁴tP₁ + 10(1-t)³t²P₂ + 10(1-t)²t³P₃ + 5(1-t)t⁴P₄ + t⁵P₅
    control_points: Tensor [6, 2] in normalized coordinates ([0,1])
    Returns: Tensor [num_samples, 2] (x, y) in pixel space
    """
    t = torch.linspace(0, 1, num_samples).unsqueeze(1).to(control_points.device)  # [num_samples, 1]

    # Quintic Bézier basis (6 control points)
    B = (1 - t) ** 5 * control_points[0] \
        + 5 * (1 - t) ** 4 * t * control_points[1] \
        + 10 * (1 - t) ** 3 * t ** 2 * control_points[2] \
        + 10 * (1 - t) ** 2 * t ** 3 * control_points[3] \
        + 5 * (1 - t) * t ** 4 * control_points[4] \
        + t ** 5 * control_points[5]

    # Scale back to pixel coordinates
    B[:, 0] = B[:, 0] * image_width
    B[:, 1] = B[:, 1] * image_height
    return B  # [num_samples, 2]

def bezier_sample_4pts(control_points, num_samples=56, image_height=720, image_width=1280):
    """
    Sample a cubic Bézier curve with 4 control points.
    control_points: Tensor [4, 2] in normalized coordinates ([0,1])
    Returns: Tensor [num_samples, 2] (x, y) in pixel space
    DEPRECATED: Use bezier_sample_6pts for quintic curves
    """
    t = torch.linspace(0, 1, num_samples).unsqueeze(1).to(control_points.device)  # [num_samples, 1]

    # Cubic Bézier basis
    B = (1 - t) ** 3 * control_points[0] \
        + 3 * (1 - t) ** 2 * t * control_points[1] \
        + 3 * (1 - t) * t ** 2 * control_points[2] \
        + t ** 3 * control_points[3]

    # Scale back to pixel coordinates
    B[:, 0] = B[:, 0] * image_width
    B[:, 1] = B[:, 1] * image_height
    return B  # [num_samples, 2]

def fit_bezier_4pts(points, image_height=720, image_width=1280):
    """
    Fit cubic Bézier (4 control points) to GT lane points.
    points: Nx2 array of (x, y) in pixel coordinates
    Returns: Tensor [4, 2] in normalized coordinates
    """
    # Normalize to [0,1]
    x = points[:, 0] / image_width
    y = points[:, 1] / image_height

    t = np.linspace(0, 1, len(points))

    def bezier_curve(ctrl):
        ctrl = ctrl.reshape(4, 2)
        B = (1 - t)[:, None] ** 3 * ctrl[0] \
            + 3 * (1 - t)[:, None] ** 2 * t[:, None] * ctrl[1] \
            + 3 * (1 - t)[:, None] * t[:, None] ** 2 * ctrl[2] \
            + t[:, None] ** 3 * ctrl[3]
        return B

    def residual(ctrl):
        pred = bezier_curve(ctrl)
        return (pred - np.stack([x, y], axis=1)).ravel()

    # Initialize control points linearly
    init_ctrl = np.stack([
        np.linspace(x[0], x[-1], 4),
        np.linspace(y[0], y[-1], 4)
    ], axis=1).ravel()

    res = least_squares(residual, init_ctrl)
    ctrl = res.x.reshape(4, 2)
    return torch.tensor(ctrl, dtype=torch.float32)

def process_tusimple_json(json_file, image_height=720, image_width=1280):
    gt_beziers = []
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f]

    for item in data:
        h_samples = np.array(item["h_samples"])
        for lane_x in item["lanes"]:
            lane_x = np.array(lane_x)
            valid_mask = lane_x > 0
            if valid_mask.sum() < 4:
                continue
            pts = np.stack([lane_x[valid_mask], h_samples[valid_mask]], axis=1)
            ctrl_pts = fit_bezier_4pts(pts, image_height, image_width)
            gt_beziers.append(ctrl_pts)
    return gt_beziers  # list of tensors [4,2]
