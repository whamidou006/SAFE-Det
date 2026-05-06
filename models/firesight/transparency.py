"""
Transparency Modeling Module.

Smoke is semi-transparent — it partially reveals the background.
Standard feature extractors struggle with this because:
1. Features contain mixed bg+smoke information
2. Smoke opacity varies spatially (dense center, thin edges)

This module models transparency by computing the difference between
features and their spatially-smoothed version. The difference highlights
what's "different" from the local context — exactly where smoke is.

Inspired by background subtraction but operating in feature space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransparencyModule(nn.Module):
    """
    Feature-level transparency modeling.

    Computes: enhanced = x + alpha * (x - smooth(x))

    The subtraction highlights regions that differ from their local context,
    which is exactly what semi-transparent smoke does to the visual scene.

    Multi-scale smoothing captures both thin wispy smoke (small kernel)
    and dense smoke columns (large kernel).

    Args:
        channels: number of feature channels
        num_scales: number of smoothing scales
        learnable_alpha: whether alpha is learned or fixed
    """

    def __init__(self, channels: int, num_scales: int = 3, learnable_alpha: bool = True):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales

        # Multi-scale Gaussian-like smoothing (depthwise conv)
        self.smoothers = nn.ModuleList()
        for i in range(num_scales):
            kernel_size = 3 + 4 * i  # 3, 7, 11
            self.smoothers.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels)
            ))

        # Scale fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True)
        )

        # Learnable blending factor
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            self.alpha = 0.5

        # Spatial attention on transparency features
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 7, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features

        Returns:
            (B, C, H, W) transparency-enhanced features
        """
        # Multi-scale background estimation
        diffs = []
        for smoother in self.smoothers:
            bg_estimate = smoother(x)
            diff = x - bg_estimate  # What's different from local context
            diffs.append(diff)

        # Fuse multi-scale transparency signals
        multi_diff = torch.cat(diffs, dim=1)
        transparency_feat = self.fusion(multi_diff)

        # Spatial gating — focus on regions with transparency signal
        spatial_mask = self.spatial_gate(transparency_feat)

        # Blend
        alpha = torch.sigmoid(self.alpha) if isinstance(self.alpha, nn.Parameter) else self.alpha
        return x + alpha * transparency_feat * spatial_mask
