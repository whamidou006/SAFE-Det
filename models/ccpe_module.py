"""
Cross Contrast Patch Embedding (CCPE) Module
Reimplemented from Wang et al. (IJIS 2025) as standalone PyTorch module.

The CCPE captures multi-scale spatial gradient information by computing
pixel differences at multiple shift distances in both horizontal and
vertical directions. This is critical for smoke detection where edges
are soft and diffuse.
"""

import torch
import torch.nn as nn


class HorizontalContrast(nn.Module):
    """Compute horizontal spatial contrasts at multiple scales."""

    def __init__(self, in_channels: int, feat_channels: int = 1,
                 steps: list = None):
        super().__init__()
        if steps is None:
            steps = [1, 2, 4, 8, 16, 32, 64, 128]
        self.steps = steps
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, feat_channels, 3, 1, 1),
                nn.Mish(inplace=True)
            ) for _ in steps
        ])
        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels + len(steps) * feat_channels, in_channels, 3, 1, 1),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_list = [x]
        for i, step in enumerate(self.steps):
            # Circular shift difference in width dimension
            x_diff = x - torch.cat([x[..., step:], x[..., :step]], dim=3)
            x_diff = self.convs[i](x_diff)
            x_list.append(x_diff)
        x = torch.cat(x_list, dim=1)
        x = self.conv_last(x)
        return x


class VerticalContrast(nn.Module):
    """Compute vertical spatial contrasts at multiple scales."""

    def __init__(self, in_channels: int, feat_channels: int = 1,
                 steps: list = None):
        super().__init__()
        if steps is None:
            steps = [1, 2, 4, 8, 16, 32, 64, 128]
        self.steps = steps
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, feat_channels, 3, 1, 1),
                nn.Mish(inplace=True)
            ) for _ in steps
        ])
        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels + len(steps) * feat_channels, in_channels, 3, 1, 1),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_list = [x]
        for i, step in enumerate(self.steps):
            # Circular shift difference in height dimension
            x_diff = x - torch.cat([x[..., step:, :], x[..., :step, :]], dim=2)
            x_diff = self.convs[i](x_diff)
            x_list.append(x_diff)
        x = torch.cat(x_list, dim=1)
        x = self.conv_last(x)
        return x


class CrossContrastPatchEmbed(nn.Module):
    """
    Cross Contrast Patch Embedding.

    Replaces standard patch embedding with contrast-enhanced version:
    1. Project input to embed_dims//2 channels via conv
    2. Apply HorizontalContrast → embed_dims//2
    3. Apply VerticalContrast → embed_dims//2
    4. Concatenate H and V contrast → embed_dims
    5. LayerNorm

    This captures multi-scale edge/gradient information critical for
    detecting soft smoke boundaries.
    """

    def __init__(self, in_channels: int = 3, embed_dims: int = 96,
                 patch_size: int = 4, stride: int = 4,
                 contrast_steps: list = None, feat_channels: int = 1):
        super().__init__()
        self.embed_dims = embed_dims

        # Initial projection to half embed_dims
        self.projection = nn.Conv2d(
            in_channels, embed_dims // 2,
            kernel_size=patch_size, stride=stride, padding=0
        )

        # Contrast modules
        self.contrast_h = HorizontalContrast(
            embed_dims // 2, feat_channels, contrast_steps
        )
        self.contrast_v = VerticalContrast(
            embed_dims // 2, feat_channels, contrast_steps
        )

        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) input image

        Returns:
            x: (B, H'*W', embed_dims) patch tokens
            hw_shape: (H', W') spatial dimensions
        """
        # Pad to patch_size multiples if needed
        B, C, H, W = x.shape
        # Project to half dimensions
        x = self.projection(x)  # (B, embed_dims//2, H', W')

        # Apply directional contrasts
        x_h = self.contrast_h(x)  # (B, embed_dims//2, H', W')
        x_v = self.contrast_v(x)  # (B, embed_dims//2, H', W')

        # Concatenate horizontal and vertical
        x = torch.cat([x_h, x_v], dim=1)  # (B, embed_dims, H', W')

        hw_shape = (x.shape[2], x.shape[3])

        # Flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dims)
        x = self.norm(x)

        return x, hw_shape
