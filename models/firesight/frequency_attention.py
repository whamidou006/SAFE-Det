"""
Frequency Attention Module (FAM).

Smoke has a distinctive frequency signature:
- Low frequency: diffuse opacity (smoke body)
- Mid frequency: gradual gradients (smoke edges)
- High frequency: sharp transitions (fire flames, not smoke)

This module decomposes features into frequency bands and applies
channel attention based on frequency content, allowing the network
to focus on the frequency ranges most relevant for each class.

Inspired by FcaNet (Qin et al., ICCV 2021) but adapted for
fire/smoke detection with explicit band separation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyAttentionModule(nn.Module):
    """
    Frequency-domain channel attention for smoke/fire features.

    Decomposes spatial features using DCT-like basis functions,
    then reweights channels based on their frequency content.

    Args:
        channels: number of input channels
        num_freq_bands: number of frequency bands to decompose into
        reduction: channel reduction ratio for attention
    """

    def __init__(self, channels: int, num_freq_bands: int = 4, reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.num_freq_bands = num_freq_bands

        # Frequency decomposition via depthwise convolutions at different scales
        self.freq_extractors = nn.ModuleList()
        for i in range(num_freq_bands):
            kernel_size = 2 * (2 ** i) + 1  # 3, 5, 9, 17
            kernel_size = min(kernel_size, 17)  # cap at 17
            self.freq_extractors.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels)
            ))

        # Channel attention from frequency descriptors
        self.band_attention = nn.Sequential(
            nn.Linear(channels * num_freq_bands, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        # Frequency band weighting (learnable)
        self.band_weights = nn.Parameter(torch.ones(num_freq_bands) / num_freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features

        Returns:
            (B, C, H, W) frequency-attended features
        """
        B, C, H, W = x.shape

        # Sequentially smooth x with progressively larger kernels.
        # band_i = smoothed_{i-1} - smoothed_i  → captures the energy in
        # the band between two scales. The first band is the *high*-pass
        # residual, the last is the smoothest low-pass.
        bands = []
        prev = x
        for extractor in self.freq_extractors:
            smoothed = extractor(prev)
            bands.append(prev - smoothed)
            prev = smoothed
        # The remaining smoothest signal is the lowest band.
        bands.append(prev)

        # Take exactly num_freq_bands bands for the fusion + attention.
        bands = bands[:self.num_freq_bands]

        # Pool each band for attention computation
        band_features = [
            F.adaptive_avg_pool2d(b.abs(), 1).squeeze(-1).squeeze(-1)
            for b in bands
        ]

        # Concatenate and compute attention
        freq_descriptor = torch.cat(band_features, dim=1)  # (B, C*num_bands)
        channel_attn = self.band_attention(freq_descriptor)  # (B, C)
        channel_attn = channel_attn.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # Weight frequency bands and sum
        band_weights = F.softmax(self.band_weights, dim=0)
        output = sum(w * b for w, b in zip(band_weights, bands))

        # Apply channel attention
        return x + output * channel_attn
