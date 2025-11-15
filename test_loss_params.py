"""
Test script to check if loss function parameters are being optimized.
"""

import torch
from losses import BezierLaneUncertaintyLoss
from arch import LaneNet

print("=" * 60)
print("Testing Loss Function Parameter Optimization")
print("=" * 60)

# Create model and loss
model = LaneNet(max_lanes=6)
criterion = BezierLaneUncertaintyLoss()

print("\n1. Loss Function Parameters:")
print("-" * 60)
for name, param in criterion.named_parameters():
    print(f"   {name}: {param.data.item():.6f} (requires_grad={param.requires_grad})")

print("\n2. Checking Optimizer Setup:")
print("-" * 60)

# Current setup (WRONG)
optimizer_wrong = torch.optim.AdamW(model.parameters(), lr=1e-4)
print(f"❌ WRONG: optimizer = AdamW(model.parameters())")
print(f"   Number of parameter groups: {len(optimizer_wrong.param_groups)}")
print(f"   Total parameters being optimized: {sum(p.numel() for group in optimizer_wrong.param_groups for p in group['params'])}")

# Check if loss params are included
loss_params_in_optimizer = False
for group in optimizer_wrong.param_groups:
    for p in group['params']:
        if p.data_ptr() == criterion.log_var_reg.data_ptr():
            loss_params_in_optimizer = True
            break

print(f"   Loss parameters included? {loss_params_in_optimizer}")

# Correct setup
print(f"\n✅ CORRECT: optimizer = AdamW(model.parameters() + criterion.parameters())")
all_params = list(model.parameters()) + list(criterion.parameters())
optimizer_correct = torch.optim.AdamW(all_params, lr=1e-4)
print(f"   Number of parameter groups: {len(optimizer_correct.param_groups)}")
print(f"   Total parameters being optimized: {sum(p.numel() for group in optimizer_correct.param_groups for p in group['params'])}")

# Check if loss params are included
loss_params_in_optimizer = False
for group in optimizer_correct.param_groups:
    for p in group['params']:
        if p.data_ptr() == criterion.log_var_reg.data_ptr():
            loss_params_in_optimizer = True
            break

print(f"   Loss parameters included? {loss_params_in_optimizer}")

print("\n3. Simulating Training:")
print("-" * 60)

# Create dummy data
dummy_input = torch.randn(2, 3, 720, 1280)
dummy_gt_ctrl = torch.rand(2, 6, 6, 2)
dummy_gt_exist = torch.randint(0, 2, (2, 6)).float()

print(f"Initial uncertainty parameters:")
print(f"   log_var_reg:   {criterion.log_var_reg.item():.6f} → σ_reg = {torch.exp(criterion.log_var_reg).sqrt().item():.6f}")
print(f"   log_var_exist: {criterion.log_var_exist.item():.6f} → σ_exist = {torch.exp(criterion.log_var_exist).sqrt().item():.6f}")
print(f"   log_var_curv:  {criterion.log_var_curv.item():.6f} → σ_curv = {torch.exp(criterion.log_var_curv).sqrt().item():.6f}")

# Training step with CORRECT optimizer
for step in range(5):
    model.train()
    outputs = model(dummy_input)
    loss_dict = criterion(outputs["bezier_refine"], dummy_gt_ctrl, 
                          outputs["exist_logits"], dummy_gt_exist)
    loss = loss_dict["total"]
    
    optimizer_correct.zero_grad()
    loss.backward()
    optimizer_correct.step()
    
    if step == 0:
        print(f"\n   After step {step + 1}:")
        print(f"      log_var_reg:   {criterion.log_var_reg.item():.6f} → σ_reg = {torch.exp(criterion.log_var_reg).sqrt().item():.6f}")
        print(f"      log_var_exist: {criterion.log_var_exist.item():.6f} → σ_exist = {torch.exp(criterion.log_var_exist).sqrt().item():.6f}")
        print(f"      log_var_curv:  {criterion.log_var_curv.item():.6f} → σ_curv = {torch.exp(criterion.log_var_curv).sqrt().item():.6f}")

print(f"\n   After step 5:")
print(f"      log_var_reg:   {criterion.log_var_reg.item():.6f} → σ_reg = {torch.exp(criterion.log_var_reg).sqrt().item():.6f}")
print(f"      log_var_exist: {criterion.log_var_exist.item():.6f} → σ_exist = {torch.exp(criterion.log_var_exist).sqrt().item():.6f}")
print(f"      log_var_curv:  {criterion.log_var_curv.item():.6f} → σ_curv = {torch.exp(criterion.log_var_curv).sqrt().item():.6f}")

# Check if parameters changed
params_changed = (
    criterion.log_var_reg.item() != 0.0 or 
    criterion.log_var_exist.item() != 0.0 or 
    criterion.log_var_curv.item() != 0.0
)

print("\n" + "=" * 60)
if params_changed:
    print("✅ SUCCESS: Uncertainty parameters are being updated!")
else:
    print("❌ FAILURE: Uncertainty parameters are NOT being updated!")
print("=" * 60)

print("\n4. The Fix:")
print("-" * 60)
print("""
In train.py, change:

    # WRONG
    optimizer = torch.optim.AdamW(model.parameters(), ...)

To:

    # CORRECT
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
""")
