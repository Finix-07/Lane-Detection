
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

# === import your components ===
from dataset_loader import dataloaders
from arch import LaneNet
from losses import BezierLaneUncertaintyLoss

# =====================================================
# 1. Configuration for Quick Testing
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

CONFIG = {
    "data_root": "data/tusimple",
    "img_size": (720, 1280),
    "batch_size": 4,
    "epochs": 10,  # Reduced for quick testing
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "val_split": 0.1,
    "save_dir": "checkpoints_quick",
    "save_freq": 2,
    "subset_ratio": 0.1,  # Use only 10% of data
}

def get_subset_loaders():
    """Create data loaders with only 10% of the dataset"""
    train_loader, val_loader = dataloaders()
    
    # Get the datasets from the loaders
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # Calculate subset sizes
    train_size = int(len(train_dataset) * CONFIG["subset_ratio"])
    val_size = int(len(val_dataset) * CONFIG["subset_ratio"])
    
    # Create random indices
    train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
    val_indices = np.random.choice(len(val_dataset), val_size, replace=False)
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create new loaders
    train_loader_subset = DataLoader(
        train_subset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader_subset = DataLoader(
        val_subset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"📊 Quick Test Mode:")
    print(f"   Train samples: {len(train_subset)} ({CONFIG['subset_ratio']*100}% of {len(train_dataset)})")
    print(f"   Val samples: {len(val_subset)} ({CONFIG['subset_ratio']*100}% of {len(val_dataset)})")
    
    return train_loader_subset, val_loader_subset

def setup_model():
    model = LaneNet()
    model = model.to(DEVICE)

    criterion = BezierLaneUncertaintyLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    return model, criterion, optimizer, scheduler

def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_exist_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, targets in pbar:
        images = images.to(DEVICE)
        gt_ctrl = targets["bezier_ctrl"].to(DEVICE)
        gt_exist = targets["lane_exist"].to(DEVICE) if "lane_exist" in targets else None

        outputs = model(images)
        pred_ctrl = outputs["bezier_refine"]
        pred_exist = outputs["exist_logits"]

        loss_dict = criterion(pred_ctrl, gt_ctrl, pred_exist, gt_exist)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reg_loss += loss_dict["reg_loss"].item()
        total_exist_loss += loss_dict["exist_loss"].item()
        
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
        pred_ctrl = outputs["bezier_refine"]
        pred_exist = outputs["exist_logits"]

        loss_dict = criterion(pred_ctrl, gt_ctrl, pred_exist, gt_exist)
        loss = loss_dict["total"]

        total_loss += loss.item()
        total_reg_loss += loss_dict["reg_loss"].item()
        total_exist_loss += loss_dict["exist_loss"].item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'σ_reg': f'{loss_dict["sigma_reg"]:.3f}',
            'σ_exist': f'{loss_dict["sigma_exist"]:.3f}'
        })

    return total_loss / len(loader), total_reg_loss / len(loader), total_exist_loss / len(loader)

def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    print("🚀 Starting Quick Test Training...")
    train_loader, val_loader = get_subset_loaders()
    model, criterion, optimizer, scheduler = setup_model()

    best_val_loss = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_reg, train_exist = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_reg, val_exist = validate(model, val_loader, criterion, epoch)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} (reg={train_reg:.4f}, exist={train_exist:.4f})  "
              f"val_loss={val_loss:.4f} (reg={val_reg:.4f}, exist={val_exist:.4f})")

        if epoch % CONFIG["save_freq"] == 0 or val_loss < best_val_loss:
            ckpt_path = os.path.join(CONFIG["save_dir"], f"lane_quick_epoch{epoch}.pth")
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

    print(f"\n🎉 Quick test complete! Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()