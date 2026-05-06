"""
Temporal Motion-Guided Fusion.

Smoke detection benefits enormously from temporal context:
- Smoke grows over time (60s–360s intervals in our dataset)
- Background is (mostly) static
- Clouds move slowly and uniformly; smoke expands from a source

This module performs cross-frame attention guided by a motion/change map,
focusing computation on regions where temporal change indicates smoke growth.

Unlike CCPE's simple 6-channel concatenation, we use:
1. Feature-level fusion (not pixel-level)
2. Motion-guided attention (focus on changed regions)
3. Lightweight design (only 2 attention layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMotionFusion(nn.Module):
    """
    Cross-frame temporal fusion guided by motion map.

    Args:
        channels: feature channel dimension
        num_heads: attention heads for cross-attention
        num_layers: number of cross-attention layers
        dropout: attention dropout rate
    """

    def __init__(self, channels: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.channels = channels

        # Motion/change detection
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
            )
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(channels, channels * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels * 2, channels),
                nn.Dropout(dropout)
            ))
            self.norms1.append(nn.LayerNorm(channels))
            self.norms2.append(nn.LayerNorm(channels))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

        # Gate for residual connection
        self.temporal_gate = nn.Parameter(torch.zeros(1))

    def forward(self, current_feat: torch.Tensor,
                prev_feat: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            current_feat: (B, C, H, W) current frame features
            prev_feat: (B, C, H, W) previous frame features (optional)

        Returns:
            (B, C, H, W) temporally-enhanced features
        """
        if prev_feat is None:
            return current_feat

        B, C, H, W = current_feat.shape

        # Compute motion map
        motion_input = torch.cat([current_feat, prev_feat], dim=1)
        motion_map = self.motion_encoder(motion_input)  # (B, 1, H, W)

        # Flatten spatial dims for attention
        curr_flat = current_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        prev_flat = prev_feat.flatten(2).transpose(1, 2)  # (B, HW, C)

        # Apply motion weighting to queries (focus on changing regions)
        motion_weight = motion_map.flatten(2).transpose(1, 2)  # (B, HW, 1)
        query = curr_flat * motion_weight  # Emphasize motion regions

        # Cross-attention: current attends to previous
        x = query
        for i in range(len(self.cross_attn_layers)):
            # Cross attention
            residual = x
            x = self.norms1[i](x)
            x, _ = self.cross_attn_layers[i](x, prev_flat, prev_flat)
            x = residual + x

            # FFN
            residual = x
            x = self.norms2[i](x)
            x = residual + self.ffn_layers[i](x)

        # Reshape back to spatial
        temporal_feat = x.transpose(1, 2).reshape(B, C, H, W)
        temporal_feat = self.output_proj(temporal_feat)

        # Gated residual (starts at 0, learns to incorporate temporal)
        gate = torch.sigmoid(self.temporal_gate)
        return current_feat + gate * temporal_feat
