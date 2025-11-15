import torch
import torch.nn as nn
import torch.nn.functional as F

class BezierLaneUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # log(σ^2) parameters — initialized to 0 (σ = 1)
        # IMPORTANT: These are learnable parameters that must be included in the optimizer!
        # optimizer = AdamW(list(model.parameters()) + list(criterion.parameters()), ...)
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_exist = nn.Parameter(torch.zeros(1))
        self.log_var_curv = nn.Parameter(torch.zeros(1))

        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.exist_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred_ctrl, gt_ctrl, pred_exist, gt_exist, lane_mask=None):
        # Regression
        reg_l = self.reg_loss_fn(pred_ctrl, gt_ctrl).mean(dim=(-1, -2))  # [B, N]
        if lane_mask is not None:
            reg_l = reg_l * lane_mask.float()
            reg_loss = reg_l.sum() / (lane_mask.sum() + 1e-6)
        else:
            reg_loss = reg_l.mean()

        # Existence
        exist_loss = self.exist_loss_fn(pred_exist.squeeze(-1), gt_exist.squeeze(-1))

        # Curvature
        delta1 = pred_ctrl[:, :, 1] - pred_ctrl[:, :, 0]
        delta2 = pred_ctrl[:, :, 2] - pred_ctrl[:, :, 1]
        delta3 = pred_ctrl[:, :, 3] - pred_ctrl[:, :, 2]
        curvature = (delta3 - 2 * delta2 + delta1).pow(2).mean()

        # Uncertainty-weighted combination
        total_loss = (
            torch.exp(-self.log_var_reg) * reg_loss * 0.5 +
            torch.exp(-self.log_var_exist) * exist_loss * 0.5 +
            torch.exp(-self.log_var_curv) * curvature * 0.5 +
            0.5 * (self.log_var_reg + self.log_var_exist + self.log_var_curv)
        )

        loss_dict = {
            "total": total_loss,
            "reg_loss": reg_loss,
            "exist_loss": exist_loss,
            "curv_loss": curvature,
            "sigma_reg": torch.exp(self.log_var_reg).item() ** 0.5,
            "sigma_exist": torch.exp(self.log_var_exist).item() ** 0.5,
            "sigma_curv": torch.exp(self.log_var_curv).item() ** 0.5,
        }

        return loss_dict
