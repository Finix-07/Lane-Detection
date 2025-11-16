from transformers import SegformerFeatureExtractor, SegformerModel
import torch
import torch.nn as nn
import torch.nn.functional as F

# Quintic Bézier Curve (6 control points):
# B(t) = (1-t)⁵P₀ + 5(1-t)⁴tP₁ + 10(1-t)³t²P₂ + 10(1-t)²t³P₃ + 5(1-t)t⁴P₄ + t⁵P₅
# where t ∈ [0,1]

# ---------------------------
# Basic building blocks
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)  # Changed to False for safer autograd
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ---------------------------
# Conv Stem
# ---------------------------
class ConvStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, 3, 2, 1),
            ConvBNReLU(32, 32, 3, 1, 1),
            ConvBNReLU(32, 64, 3, 2, 1)
        )
    def forward(self, x):
        return self.stem(x)  # (B,64,H/4,W/4)

# ---------------------------
# Shallow CNN Stage
# ---------------------------
class ShallowCNNStage(nn.Module):
    def __init__(self, in_ch=64, out_ch=128):
        super().__init__()
        layers = []
        # first block: change channels from in_ch -> out_ch
        layers.append(nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1)
        ))
        # remaining blocks keep channels at out_ch
        for _ in range(2):
            layers.append(nn.Sequential(
                ConvBNReLU(out_ch, out_ch, 3, 1, 1),
                ConvBNReLU(out_ch, out_ch, 3, 1, 1)
            ))
        self.blocks = nn.Sequential(*layers)
    def forward(self, x):
        return self.blocks(x)

# ---------------------------
# Conv Adapter + FPN Fusion
# ---------------------------
class ConvAdapterFPN(nn.Module):
    def __init__(self, in_dims=[64, 160, 256], out_dim=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, 1) for in_dim in in_dims
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, 3, 1, 1) for _ in in_dims
        ])

    def forward(self, c2, c3, c4):
        # 1×1 conv to align channels
        p4 = self.lateral_convs[2](c4)
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)

        # smoothing
        p4 = self.smooth_convs[2](p4)
        p3 = self.smooth_convs[1](p3)
        p2 = self.smooth_convs[0](p2)

        return p2, p3, p4


# ---------------------------
# MiT Backbone
# ---------------------------

class MiTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mit = SegformerModel.from_pretrained("nvidia/mit-b0")
        self.fpn = ConvAdapterFPN(in_dims=[64, 160, 256], out_dim=128)

    def forward(self, x):
        # x must be normalized RGB image tensor (B,3,H,W)
        # Expected: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        outputs = self.mit(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        # Extract multi-scale features (indices validated for mit-b0)
        c2, c3, c4 = hidden_states[1], hidden_states[2], hidden_states[3]
        # Apply FPN fusion and return fused features
        p2, p3, p4 = self.fpn(c2, c3, c4)
        return p2, p3, p4

# ---------------------------
# RESA+ (directional propagation)
# ---------------------------
class RESAPlus(nn.Module):
    def __init__(self, ch=128, iter_steps=4, kernel_size=9, alpha=0.5):
        """
        ch: input/output channel dimension
        iter_steps: number of propagation iterations
        kernel_size: 1D conv kernel size for directional propagation
        alpha: scaling factor for aggregation strength
        """
        super().__init__()
        self.iter_steps = iter_steps
        self.alpha = alpha

        # Directional 1D convs (depthwise)
        self.conv_left = nn.Conv2d(ch, ch, kernel_size=(1, kernel_size),
                                   stride=1, padding=(0, kernel_size // 2),
                                   groups=ch, bias=False)
        self.conv_right = nn.Conv2d(ch, ch, kernel_size=(1, kernel_size),
                                    stride=1, padding=(0, kernel_size // 2),
                                    groups=ch, bias=False)
        self.conv_up = nn.Conv2d(ch, ch, kernel_size=(kernel_size, 1),
                                 stride=1, padding=(kernel_size // 2, 0),
                                 groups=ch, bias=False)
        self.conv_down = nn.Conv2d(ch, ch, kernel_size=(kernel_size, 1),
                                   stride=1, padding=(kernel_size // 2, 0),
                                   groups=ch, bias=False)

        # Learnable gate for combining directional messages
        self.gate = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.Sigmoid()
        )

        self.norm = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=False)  # Changed to False - safer with residual connections

    def forward(self, x):
        feat = x
        for _ in range(self.iter_steps):
            # Directional message passing
            left = self.conv_left(feat)
            right = self.conv_right(feat)
            up = self.conv_up(feat)
            down = self.conv_down(feat)

            # Combine directions
            agg = (left + right + up + down) / 4.0
            gate = self.gate(feat)
            feat = feat + self.alpha * gate * agg

            # Normalization + activation
            feat = self.act(self.norm(feat))
        return feat

# ---------------------------
# Strip-based Proposal Head
# ---------------------------
class StripProposalHead(nn.Module):
    def __init__(self, in_ch=128, num_strips=72, use_offset=True):
        super().__init__()
        self.num_strips = num_strips
        self.use_offset = use_offset

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        # Predict confidence per strip (lane existence)
        self.conf_head = nn.Conv2d(64, num_strips, 1)

        # Optional offset regression per strip (y offset)
        if use_offset:
            self.offset_head = nn.Conv2d(64, num_strips, 1)
        else:
            self.offset_head = None

    def forward(self, x):
        feat = self.conv(x)
        conf = self.conf_head(feat)  # [B, num_strips, H, W]
        offset = self.offset_head(feat) if self.offset_head else None
        return {"conf": conf, "offset": offset}

# ---------------------------
# Auxiliary Segmentation Head
# ---------------------------
class SegmentationHead(nn.Module):
    def __init__(self, in_ch=128):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 1)
        )
    def forward(self, x, target_size):
        # target_size: (H, W) tuple for output size
        out = torch.sigmoid(self.conv(x))
        return F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

# ---------------------------
# Bézier Regression Heads (Quintic - 6 control points, multi-lane)
# ---------------------------
class BezierCoarseHead(nn.Module):
    def __init__(self, in_ch=128, num_ctrl=6, max_lanes=6):
        super().__init__()
        self.num_ctrl = num_ctrl
        self.max_lanes = max_lanes
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(in_ch, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(128, max_lanes * num_ctrl * 2),  # (x, y) for each control point per lane
            nn.Sigmoid()  # Apply sigmoid directly in the architecture
        )

    def forward(self, feat):
        # feat: [B, in_ch, H, W]
        pooled = self.pool(feat).flatten(1)  # [B, in_ch]
        out = self.regressor(pooled)
        return out.view(-1, self.max_lanes, self.num_ctrl, 2)  # [B, max_lanes, num_ctrl, 2]

class BezierRefineHead(nn.Module):
    def __init__(self, in_ch=128, num_ctrl=6, max_lanes=6):
        super().__init__()
        self.num_ctrl = num_ctrl
        self.max_lanes = max_lanes
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.refine = nn.Sequential(
            nn.Linear(in_ch + max_lanes * num_ctrl * 2, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, max_lanes * num_ctrl * 2),
            nn.Tanh()  # Output in [-1, 1] for delta
        )

    def forward(self, feat, coarse_pts):
        # feat: [B, in_ch, H, W], coarse_pts: [B, max_lanes, num_ctrl, 2]
        pooled = self.pool(feat).flatten(1)  # [B, in_ch]
        # Ensure coarse_pts has same dtype/device as pooled for safe concatenation
        coarse_flat = coarse_pts.flatten(1).to(dtype=pooled.dtype, device=pooled.device)
        feat_flat = torch.cat([pooled, coarse_flat], dim=1)
        delta = self.refine(feat_flat).view(-1, self.max_lanes, self.num_ctrl, 2)
        # Scale delta to reasonable range (e.g., ±0.2 of image size)
        delta = delta * 0.2
        refined = coarse_pts + delta
        # Clamp instead of sigmoid to avoid gradient saturation
        return torch.clamp(refined, 0.0, 1.0)

# ---------------------------
# Lane Existence Head
# ---------------------------
class ExistenceHead(nn.Module):
    def __init__(self, in_ch=128, num_lanes=6):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, num_lanes)  # logits for each lane
        )
    
    def forward(self, feat):
        # feat: [B, in_ch, H, W]
        pooled = self.pool(feat).flatten(1)  # [B, in_ch]
        return self.fc(pooled)  # [B, num_lanes]

# ---------------------------
# Complete Network Wrapper
# ---------------------------
class LaneNet(nn.Module):
    def __init__(self, max_lanes=6):
        super().__init__()
        # MiTBackbone now returns FPN outputs directly
        self.mit = MiTBackbone()
        self.resa = RESAPlus(ch=128, iter_steps=4, kernel_size=9)
        self.prop_head = StripProposalHead()
        self.seg_head = SegmentationHead()
        self.coarse = BezierCoarseHead(num_ctrl=6, max_lanes=max_lanes)
        self.refine = BezierRefineHead(num_ctrl=6, max_lanes=max_lanes)
        self.exist_head = ExistenceHead(in_ch=128, num_lanes=max_lanes)

    def forward(self, x):
        # x: [B, 3, H, W] - Expected to be normalized by dataset loader
        # with mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        img = x
        input_size = img.shape[-2:]  # (H, W)
        
        # Get FPN features from MiT backbone
        p2, p3, p4 = self.mit(img)
        
        # Apply RESA+ for spatial relationship modeling
        p3 = self.resa(p3)
        
        # Multi-task heads
        proposals = self.prop_head(p3)
        seg = self.seg_head(p3, target_size=input_size)  # Pass target size
        coarse = self.coarse(p3)
        refine = self.refine(p3, coarse)
        exist = self.exist_head(p3)
        
        return {
            'proposals': proposals,
            'segmentation': seg,
            'bezier_coarse': coarse,
            'bezier_refine': refine,
            'exist_logits': exist
        }

# ---------------------------
# Test dummy forward
# ---------------------------
if __name__ == '__main__':
    model = LaneNet()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params} ({total_params/1e6:.2f}M)")