"""
Deformable Contrast Module (DCM) — Evolution of CCPE.

Instead of fixed pixel shifts at [1,2,4,8,16,32,64,128], learns
deformable sampling offsets that adapt to content. The network discovers
optimal spatial contexts for discriminating smoke, fire, and background.

Key advantages over CCPE:
1. Content-adaptive: offsets vary per pixel/feature
2. Directional freedom: not limited to H/V, can learn diagonal/curved
3. Fewer parameters: single deformable conv vs 16 separate conv branches
4. Better for fire: fire has sharp, irregular boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableContrastModule(nn.Module):
    """
    Deformable Contrast: learn where to look for spatial differences.

    For each spatial position, predicts K offset vectors, samples features
    at those offsets, computes differences with center feature, and fuses.

    Args:
        channels: input/output feature channels
        num_points: number of sampling points per direction (default 8)
        groups: number of offset groups (multi-head deformable)
    """

    def __init__(self, channels: int, num_points: int = 8, groups: int = 4):
        super().__init__()
        self.channels = channels
        self.num_points = num_points
        self.groups = groups

        # Offset prediction: 2 (x,y) offsets per point
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2 * num_points * groups, 3, 1, 1)
        )

        # Contrast fusion: process all differences
        self.contrast_conv = nn.Sequential(
            nn.Conv2d(channels * num_points, channels, 1, groups=groups),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups),
            nn.BatchNorm2d(channels),
        )

        # Learnable gate for residual connection
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self._init_offsets()

    def _init_offsets(self):
        """Initialize offsets to mimic CCPE's fixed shifts."""
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)
        # Initialize bias to approximate [1,2,4,8,16,32,64,128] shifts
        with torch.no_grad():
            bias = self.offset_conv[-1].bias
            shifts = [1, 2, 4, 8, 16, 32, 64, 128]
            for g in range(self.groups):
                for k, shift in enumerate(shifts[:self.num_points]):
                    idx = g * self.num_points * 2 + k * 2
                    if g % 2 == 0:  # Horizontal groups
                        bias[idx] = shift  # dx
                        bias[idx + 1] = 0  # dy
                    else:  # Vertical groups
                        bias[idx] = 0
                        bias[idx + 1] = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features

        Returns:
            (B, C, H, W) contrast-enhanced features
        """
        B, C, H, W = x.shape

        # Predict offsets
        offsets = self.offset_conv(x)  # (B, 2*K*G, H, W)
        offsets = offsets.reshape(B, self.groups, self.num_points, 2, H, W)

        # Sample at offsets and compute contrasts
        contrasts = []
        for k in range(self.num_points):
            # Get offset for this point across all groups
            offset_k = offsets[:, :, k]  # (B, G, 2, H, W)
            # Average across groups for simplicity
            offset_mean = offset_k.mean(dim=1)  # (B, 2, H, W)

            # Bilinear sampling at offset positions
            sampled = self._sample_at_offset(x, offset_mean)
            # Contrast = difference from center
            contrast = x - sampled
            contrasts.append(contrast)

        # Stack and fuse
        contrasts = torch.cat(contrasts, dim=1)  # (B, C*K, H, W)
        enhanced = self.contrast_conv(contrasts)  # (B, C, H, W)

        # Gated residual
        gate = self.gate(enhanced)
        return x + gate * enhanced

    def _sample_at_offset(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """
        Sample features at offset positions using grid_sample.

        Args:
            x: (B, C, H, W) input features
            offset: (B, 2, H, W) offset field (dx, dy in pixels)

        Returns:
            (B, C, H, W) sampled features
        """
        B, C, H, W = x.shape

        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Convert pixel offsets to normalized coordinates
        offset_norm = torch.stack([
            offset[:, 0] * 2 / W,  # dx normalized
            offset[:, 1] * 2 / H,  # dy normalized
        ], dim=-1)  # (B, H, W, 2)

        # Sample
        grid_offset = grid + offset_norm
        sampled = F.grid_sample(x, grid_offset, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        return sampled
