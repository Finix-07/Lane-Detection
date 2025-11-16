"""
Retrain Lane Detection Model with Fixed Loss Function

This script addresses the model collapse issue where predictions were identical
for all images due to broken uncertainty-weighted loss.

Key changes:
1. ✅ Simple fixed-weight loss (no learnable uncertainty)
2. ✅ Proper loss scaling to prevent negative values
3. ✅ Better optimizer configuration
4. ✅ Gradient clipping for stability
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.arch import LaneNet
from src.data.dataset_loader import TuSimpleBezierDataset
from src.models.losses import BezierLaneLoss

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Starting training on device: {DEVICE}")

# Configuration
CONFIG = {
    "batch_size": 4,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "val_split": 0.1,
    "save_dir": "checkpoints/experiments",
    "save_freq": 5,
    "grad_clip": 1.0,  # Gradient clipping for stability
    
    # Loss weights (updated for BezierLaneLossFinal)
    "w_ctrl": 1.0,      # Control point loss (coarse)
    "w_curve": 2.0,     # Sampled curve loss (fine-grained) - auto-scaled
    "w_exist": 1.0,     # Lane existence
    "w_curv": 0.05,     # Curvature smoothness
    
    # Image dimensions for loss function
    "img_width": 1280,
    "img_height": 720,
}

# Create checkpoint directory
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==================== DATA LOADING ====================
print("\n📂 Loading dataset...")
full_dataset = TuSimpleBezierDataset(split="train")

train_size = int((1 - CONFIG["val_split"]) * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], 
                         shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], 
                       shuffle=False, num_workers=0, pin_memory=True)

print(f"✅ Dataset loaded:")
print(f"   Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
print(f"   Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")

# ==================== MODEL & LOSS ====================
print("\n🏗️ Initializing model...")
model = LaneNet(max_lanes=6).to(DEVICE)

# Use improved loss with sampled curve supervision
criterion = BezierLaneLoss(
    num_ctrl=6,
    num_samples=50,
    w_ctrl=CONFIG["w_ctrl"],
    w_curve=CONFIG["w_curve"],
    w_exist=CONFIG["w_exist"],
    w_curv=CONFIG["w_curv"],
    img_width=CONFIG["img_width"],
    img_height=CONFIG["img_height"]
).to(DEVICE)

print(f"✅ Model initialized")
print(f"   Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"   Loss weights: ctrl={CONFIG['w_ctrl']}, curve={CONFIG['w_curve']} (auto-scaled), exist={CONFIG['w_exist']}, curv={CONFIG['w_curv']}")

# ==================== OPTIMIZER ====================
# NOTE: No criterion.parameters() since we removed learnable uncertainty weights
optimizer = torch.optim.AdamW(
    model.parameters(),  # Only model parameters now!
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"]
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

print(f"✅ Optimizer: AdamW (lr={CONFIG['lr']}, weight_decay={CONFIG['weight_decay']})")
print(f"✅ Scheduler: StepLR (step_size=15, gamma=0.5)")
print(f"✅ Gradient clipping: {CONFIG['grad_clip']}")

# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(model, loader, optimizer, criterion, epoch, device):
    model.train()
    total_loss = 0.0
    total_ctrl_loss = 0.0
    total_curve_loss = 0.0
    total_exist_loss = 0.0
    total_curv_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        gt_ctrl = targets["bezier_ctrl"].to(device)
        gt_exist = targets["lane_exist"].to(device)
        
        # Forward
        outputs = model(images)
        pred_ctrl = outputs["bezier_refine"]
        pred_exist = outputs["exist_logits"]
        
        # Compute loss
        loss_dict = criterion(pred_ctrl, gt_ctrl, pred_exist, gt_exist)
        loss = loss_dict["total"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if CONFIG["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_ctrl_loss += loss_dict.get("ctrl", loss_dict.get("reg_loss", 0)).item() if torch.is_tensor(loss_dict.get("ctrl", 0)) else loss_dict.get("ctrl", 0)
        total_curve_loss += loss_dict.get("curve", 0).item() if torch.is_tensor(loss_dict.get("curve", 0)) else loss_dict.get("curve", 0)
        total_exist_loss += loss_dict.get("exist", loss_dict.get("exist_loss", 0)).item() if torch.is_tensor(loss_dict.get("exist", 0)) else loss_dict.get("exist", 0)
        total_curv_loss += loss_dict.get("curv", loss_dict.get("curv_loss", 0)).item() if torch.is_tensor(loss_dict.get("curv", 0)) else loss_dict.get("curv", 0)
        
        # Update progress bar (show only key losses to keep it readable)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ctrl': f'{loss_dict.get("ctrl", loss_dict.get("reg_loss", 0)):.4f}',
            'curve': f'{loss_dict.get("curve", 0):.4f}' if "curve" in loss_dict else None,
            'exist': f'{loss_dict.get("exist", loss_dict.get("exist_loss", 0)):.4f}',
        })
        # Check for invalid loss
        if loss.item() < 0 or torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ WARNING: Invalid loss at batch {batch_idx}")
            print(f"   Loss: {loss.item()}")
            print(f"   Pred ctrl range: [{pred_ctrl.min():.3f}, {pred_ctrl.max():.3f}]")
            print(f"   GT ctrl range: [{gt_ctrl.min():.3f}, {gt_ctrl.max():.3f}]")
    
    n = len(loader)
    return total_loss / n, total_ctrl_loss / n, total_curve_loss / n, total_exist_loss / n, total_curv_loss / n


@torch.no_grad()
def validate(model, loader, criterion, epoch, device):
    model.eval()
    total_loss = 0.0
    total_ctrl_loss = 0.0
    total_curve_loss = 0.0
    total_exist_loss = 0.0
    total_curv_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Val]  ")
    for images, targets in pbar:
        images = images.to(device)
        gt_ctrl = targets["bezier_ctrl"].to(device)
        gt_exist = targets["lane_exist"].to(device)
        
        outputs = model(images)
        pred_ctrl = outputs["bezier_refine"]
        pred_exist = outputs["exist_logits"]
        
        loss_dict = criterion(pred_ctrl, gt_ctrl, pred_exist, gt_exist)
        loss = loss_dict["total"]
        
        total_loss += loss.item()
        total_ctrl_loss += loss_dict.get("ctrl", loss_dict.get("reg_loss", 0)).item() if torch.is_tensor(loss_dict.get("ctrl", 0)) else loss_dict.get("ctrl", 0)
        total_curve_loss += loss_dict.get("curve", 0).item() if torch.is_tensor(loss_dict.get("curve", 0)) else loss_dict.get("curve", 0)
        total_exist_loss += loss_dict.get("exist", loss_dict.get("exist_loss", 0)).item() if torch.is_tensor(loss_dict.get("exist", 0)) else loss_dict.get("exist", 0)
        total_curv_loss += loss_dict.get("curv", loss_dict.get("curv_loss", 0)).item() if torch.is_tensor(loss_dict.get("curv", 0)) else loss_dict.get("curv", 0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ctrl': f'{loss_dict.get("ctrl", loss_dict.get("reg_loss", 0)):.4f}',
            'curve': f'{loss_dict.get("curve", 0):.4f}' if "curve" in loss_dict else None,
            'exist': f'{loss_dict.get("exist", loss_dict.get("exist_loss", 0)):.4f}'
        })
    
    n = len(loader)
    return total_loss / n, total_ctrl_loss / n, total_curve_loss / n, total_exist_loss / n, total_curv_loss / n


# ==================== TRAINING LOOP ====================
print(f"\n{'='*80}")
print(f"STARTING TRAINING")
print(f"{'='*80}\n")

best_val_loss = float("inf")
train_losses = []
val_losses = []

for epoch in range(1, CONFIG["epochs"] + 1):
    # Train
    train_loss, train_ctrl, train_curve, train_exist, train_curv = train_one_epoch(
        model, train_loader, optimizer, criterion, epoch, DEVICE
    )
    
    # Validate
    val_loss, val_ctrl, val_curve, val_exist, val_curv = validate(
        model, val_loader, criterion, epoch, DEVICE
    )
    
    # Step scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Log results
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"\nEpoch {epoch:02d}/{CONFIG['epochs']}:")
    print(f"  Train: loss={train_loss:.4f} (ctrl={train_ctrl:.4f}, curve={train_curve:.4f}, exist={train_exist:.4f}, curv={train_curv:.4f})")
    print(f"  Val:   loss={val_loss:.4f} (ctrl={val_ctrl:.4f}, curve={val_curve:.4f}, exist={val_exist:.4f}, curv={val_curv:.4f})")
    print(f"  LR: {current_lr:.6f}")
    
    # Check for invalid loss
    if val_loss < 0:
        print(f"  ⚠️ WARNING: Negative validation loss! This should not happen with fixed loss.")
    
    # Save checkpoints
    if epoch % CONFIG["save_freq"] == 0 or val_loss < best_val_loss:
        ckpt_path = os.path.join(CONFIG["save_dir"], f"lane_epoch{epoch:02d}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": CONFIG,
        }, ckpt_path)
        print(f"  ✅ Saved → {ckpt_path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CONFIG["save_dir"], "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_loss": val_loss,
                "config": CONFIG,
            }, best_path)
            print(f"  🌟 New best model! Val loss: {val_loss:.4f}")

print(f"\n{'='*80}")
print(f"✅ TRAINING COMPLETED")
print(f"{'='*80}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final model saved to: {CONFIG['save_dir']}/best_model.pth")

# ==================== PLOT TRAINING CURVES ====================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress (Log Scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(CONFIG["save_dir"], "training_curve.png")
plt.savefig(save_path, dpi=150)
print(f"\n✅ Training curve saved to: {save_path}")
plt.close()

print(f"\n{'='*80}")
print("📊 TRAINING SUMMARY")
print(f"{'='*80}")
print(f"Total epochs:        {CONFIG['epochs']}")
print(f"Best val loss:       {best_val_loss:.4f}")
print(f"Final train loss:    {train_losses[-1]:.4f}")
print(f"Final val loss:      {val_losses[-1]:.4f}")
print(f"Checkpoints saved:   {CONFIG['save_dir']}/")
print(f"\n🎯 Next steps:")
print(f"   1. Run inference: python inference_fixed.py")
print(f"   2. Load best model from: {CONFIG['save_dir']}/best_model.pth")
print(f"   3. Check if predictions are now different for each image!")
