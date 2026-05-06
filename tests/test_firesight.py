"""Unit tests for the FireSight novel modules and full detector.

The real DINOv2 backbone is fetched from torch.hub which requires internet
access. For tests we set SAFE_DET_OFFLINE_BACKBONE=1 (in conftest.py) and
inject a tiny synthetic backbone via monkey-patching.
"""
import os
import pytest
import torch
import torch.nn as nn

from models.firesight.deformable_contrast import DeformableContrastModule
from models.firesight.frequency_attention import FrequencyAttentionModule
from models.firesight.transparency import TransparencyModule
from models.firesight.temporal_fusion import TemporalMotionFusion


# ── SAFE individual modules ────────────────────────────────────────────────

def test_dcm_shape(small_features):
    mod = DeformableContrastModule(channels=32, num_points=4, groups=2).to(
        small_features.device
    )
    out = mod(small_features)
    assert out.shape == small_features.shape


def test_dcm_backward(device):
    mod = DeformableContrastModule(channels=8, num_points=2, groups=1).to(device)
    x = torch.randn(1, 8, 16, 16, device=device, requires_grad=True)
    out = mod(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_fam_shape(small_features):
    mod = FrequencyAttentionModule(channels=32, num_freq_bands=3).to(
        small_features.device
    )
    out = mod(small_features)
    assert out.shape == small_features.shape


def test_fam_backward(device):
    mod = FrequencyAttentionModule(channels=8, num_freq_bands=2).to(device)
    x = torch.randn(1, 8, 16, 16, device=device, requires_grad=True)
    mod(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_transparency_shape_and_alpha_learnable(small_features):
    mod = TransparencyModule(channels=32, num_scales=2).to(small_features.device)
    out = mod(small_features)
    assert out.shape == small_features.shape
    assert isinstance(mod.alpha, nn.Parameter)
    assert mod.alpha.requires_grad


def test_transparency_fixed_alpha(small_features):
    mod = TransparencyModule(channels=32, num_scales=2,
                             learnable_alpha=False).to(small_features.device)
    out = mod(small_features)
    assert out.shape == small_features.shape
    assert mod.alpha == 0.5


def test_temporal_fusion_with_prev(small_features):
    mod = TemporalMotionFusion(channels=32, num_heads=4, num_layers=2).to(
        small_features.device
    )
    prev = torch.randn_like(small_features)
    out = mod(small_features, prev)
    assert out.shape == small_features.shape


def test_temporal_fusion_without_prev_passes_through(small_features):
    mod = TemporalMotionFusion(channels=32).to(small_features.device)
    out = mod(small_features, prev_feat=None)
    # When prev is None we should get the input back unchanged
    assert torch.equal(out, small_features)


# ── Full FireSightDetector with offline backbone ───────────────────────────

class _StubDinoBackbone(nn.Module):
    """Tiny ConvNeXt-ish substitute for DINOv2 used in tests so we never
    touch the network. Exposes the same (P3, P4, P5) interface and accepts
    the real backbone's kwargs."""

    def __init__(self, model_name='dinov2_vits14', out_channels=64,
                 freeze_backbone=False):
        super().__init__()
        self.out_channels = out_channels
        self.stem = nn.Conv2d(3, out_channels, 7, stride=4, padding=3)
        self.down1 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        p3 = self.stem(x)            # stride 4
        p4 = self.down1(p3)          # stride 8
        p5 = self.down2(p4)          # stride 16
        return p3, p4, p5


def test_firesight_with_stub_backbone(device, monkeypatch):
    """Construct FireSight with a stub backbone (avoids torch.hub)."""
    from models.firesight import firesight_detector as fd_mod

    # Monkey-patch the backbone class so __init__ doesn't try torch.hub.
    monkeypatch.setattr(fd_mod, "DINOv2Backbone", _StubDinoBackbone)
    model = fd_mod.FireSightDetector(
        num_classes=2,
        backbone_type="dinov2",
        backbone_channels=64,
        use_dcm=True, use_fam=True, use_tm=True, use_temporal=False,
        head_type="yolox",
        input_size=(128, 128),
    ).to(device)

    x = torch.randn(1, 3, 128, 128, device=device)

    model.train()
    cls, bbox, obj = model(x)
    assert len(cls) == 3 and len(bbox) == 3 and len(obj) == 3

    model.eval()
    with torch.no_grad():
        decoded = model(x)
    assert decoded.dim() == 3 and decoded.shape[-1] == 4 + 2


def test_firesight_temporal_runs(device, monkeypatch):
    from models.firesight import firesight_detector as fd_mod

    monkeypatch.setattr(fd_mod, "DINOv2Backbone", _StubDinoBackbone)
    model = fd_mod.FireSightDetector(
        num_classes=2,
        backbone_type="dinov2",
        backbone_channels=64,
        use_dcm=False, use_fam=False, use_tm=False, use_temporal=True,
        head_type="yolox",
        input_size=(128, 128),
    ).to(device)
    x = torch.randn(1, 3, 128, 128, device=device)
    prev = torch.randn(1, 3, 128, 128, device=device)
    model.train()
    cls, bbox, obj = model(x, prev_x=prev)
    assert len(cls) == 3
