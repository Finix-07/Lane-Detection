import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# === import your components ===
from src.data.dataset_loader import dataloaders
from src.models.arch import LaneNet  # your full model           # custom loss module
from src.models.losses import BezierLaneUncertaintyLoss

# =====================================================
# 1. Configuration
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

CONFIG = {
    "data_root": "data/tusimple",
    "img_size": (720, 1280),
    "batch_size": 4,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "val_split": 0.1,
    "save_dir": "checkpoints/experiments",
    "save_freq": 5,
}

# =====================================================
# 3. Setup Model, Loss, Optimizer
# =====================================================
def setup_model():
    model = LaneNet()  # your full architecture
    model = model.to(DEVICE)

    criterion = BezierLaneUncertaintyLoss().to(DEVICE)  # your loss for control points
    # Include both model AND criterion parameters for optimization
    # This allows the uncertainty weights (log_var_*) to be learned
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    return model, criterion, optimizer, scheduler


# =====================================================
# 4. Training + Validation Loops
# =====================================================

def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_exist_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, targets in pbar:
        images = images.to(DEVICE)
        gt_ctrl = targets["bezier_ctrl"].to(DEVICE)  # [B, num_lanes, num_ctrl, 2]
        gt_exist = targets["lane_exist"].to(DEVICE) if "lane_exist" in targets else None  # [B, num_lanes]

        outputs = model(images)
        # Use the actual keys from model output
        pred_ctrl = outputs["bezier_refine"]  # changed from "ctrl_pts"
        pred_exist = outputs["exist_logits"]

        loss_dict = criterion(pred_ctrl, gt_ctrl, pred_exist, gt_exist)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reg_loss += loss_dict["reg_loss"].item()
        total_exist_loss += loss_dict["exist_loss"].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'σ_reg': f'{loss_dict["sigma_reg"]:.3f}',
            'σ_exist': f'{loss_dict["sigma_exist"]:.3f}'
        })

    return total_loss / len(loader), total_reg_loss / len(loader), total_exist_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, epoch):
    model.eval()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_exist_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for images, targets in pbar:
        images = images.to(DEVICE)
        gt_ctrl = targets["bezier_ctrl"].to(DEVICE)
        gt_exist = targets["lane_exist"].to(DEVICE) if "lane_exist" in targets else None

        outputs = model(images)
        # Use the actual keys from model output
        pred_ctrl = outputs["bezier_refine"]  # changed from "ctrl_pts"
        pred_exist = outputs["exist_logits"]

        loss_dict = criterion(pred_ctrl, gt_ctrl, pred_exist, gt_exist)
        loss = loss_dict["total"]

        total_loss += loss.item()
        total_reg_loss += loss_dict["reg_loss"].item()
        total_exist_loss += loss_dict["exist_loss"].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'σ_reg': f'{loss_dict["sigma_reg"]:.3f}',
            'σ_exist': f'{loss_dict["sigma_exist"]:.3f}'
        })

    return total_loss / len(loader), total_reg_loss / len(loader), total_exist_loss / len(loader)


# =====================================================
# 5. Main
# =====================================================
def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    train_loader, val_loader = dataloaders()
    model, criterion, optimizer, scheduler = setup_model()

    best_val_loss = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_reg, train_exist = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_reg, val_exist = validate(model, val_loader, criterion, epoch)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} (reg={train_reg:.4f}, exist={train_exist:.4f})  "
              f"val_loss={val_loss:.4f} (reg={val_reg:.4f}, exist={val_exist:.4f})")

        # Save checkpoints
        if epoch % CONFIG["save_freq"] == 0 or val_loss < best_val_loss:
            ckpt_path = os.path.join(CONFIG["save_dir"], f"lane_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"✅ Saved checkpoint → {ckpt_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss


if __name__ == "__main__":
    main()
