"""
YOLOX-style PAFPN Neck for CCPE detector.
Path Aggregation Feature Pyramid Network with CSP bottlenecks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, kernel=1, stride=1, groups=1, act=True):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    def __init__(self, in_c, out_c, n=1, shortcut=True):
        super().__init__()
        hidden_c = out_c // 2
        self.cv1 = ConvBNAct(in_c, hidden_c, 1)
        self.cv2 = ConvBNAct(in_c, hidden_c, 1)
        self.cv3 = ConvBNAct(2 * hidden_c, out_c, 1)
        self.blocks = nn.Sequential(*[
            Bottleneck(hidden_c, hidden_c, shortcut) for _ in range(n)
        ])

    def forward(self, x):
        return self.cv3(torch.cat([self.blocks(self.cv1(x)), self.cv2(x)], dim=1))


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, shortcut=True):
        super().__init__()
        self.cv1 = ConvBNAct(in_c, out_c, 1)
        self.cv2 = ConvBNAct(out_c, out_c, 3)
        self.shortcut = shortcut and in_c == out_c

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.shortcut else out


class YOLOXPAFPN(nn.Module):
    """
    YOLOX Path Aggregation FPN.

    Takes multi-scale features from backbone (3 levels) and produces
    fused features at the same 3 scales with uniform channel count.

    Args:
        in_channels: list of input channel counts [C3, C4, C5]
        out_channels: output channel count for all levels
        num_csp_blocks: number of bottleneck blocks in CSP
    """

    def __init__(self, in_channels=(192, 384, 768), out_channels=128, num_csp_blocks=1):
        super().__init__()
        self.in_channels = in_channels

        # Lateral convs (reduce channels)
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(c, out_channels, 1) for c in in_channels
        ])

        # Top-down (FPN) path
        self.fpn_blocks = nn.ModuleList([
            CSPBlock(2 * out_channels, out_channels, num_csp_blocks, shortcut=False)
            for _ in range(len(in_channels) - 1)
        ])

        # Bottom-up (PAN) path
        self.downsample_convs = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, 3, stride=2)
            for _ in range(len(in_channels) - 1)
        ])
        self.pan_blocks = nn.ModuleList([
            CSPBlock(2 * out_channels, out_channels, num_csp_blocks, shortcut=False)
            for _ in range(len(in_channels) - 1)
        ])

    def forward(self, inputs):
        """
        Args:
            inputs: tuple of (C3, C4, C5) features from backbone

        Returns:
            tuple of 3 fused feature maps at strides (8, 16, 32)
        """
        assert len(inputs) == len(self.in_channels)

        # Lateral connections
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
            laterals[i - 1] = self.fpn_blocks[i - 1](
                torch.cat([laterals[i - 1], upsampled], dim=1)
            )

        # Bottom-up path
        outs = [laterals[0]]
        for i in range(len(laterals) - 1):
            downsampled = self.downsample_convs[i](outs[-1])
            outs.append(self.pan_blocks[i](
                torch.cat([downsampled, laterals[i + 1]], dim=1)
            ))

        return tuple(outs)
