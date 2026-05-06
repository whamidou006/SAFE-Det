"""Tests for model factory dispatch.

Verifies that train.py and eval.py honour the `model.type` field
in the config — i.e. selecting `firesight` actually instantiates the
FireSightDetector and not the CCPE baseline.
"""
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# These imports require the bug-fix described in the review (see fix in train.py).
from train import build_model  # noqa: E402  ← will fail if not implemented
from models.detector import CCPE_Detector  # noqa: E402
from models.firesight.firesight_detector import FireSightDetector  # noqa: E402


def _ccpe_cfg():
    return {
        "model": {
            "num_classes": 2, "embed_dims": 32,
            "depths": [2, 2, 2, 2], "num_heads": [2, 4, 8, 8],
            "window_size": 4, "fpn_channels": 32,
            "input_size": [128, 128],
            "contrast_steps": [1, 2, 4],
        },
    }


def _firesight_cfg():
    return {
        "model": {
            "type": "firesight",
            "num_classes": 2,
            "backbone_type": "dinov2",
            "backbone_channels": 64,
            "use_dcm": True, "use_fam": True, "use_tm": True,
            "use_temporal": False,
            "head_type": "yolox",
            "input_size": [128, 128],
        },
    }


def test_build_model_dispatches_ccpe_by_default():
    model = build_model(_ccpe_cfg())
    assert isinstance(model, CCPE_Detector)


def test_build_model_dispatches_firesight(monkeypatch):
    """Dispatch must build FireSightDetector when type=='firesight'.
    We swap the DINOv2 backbone for a stub so no network is needed."""
    from models.firesight import firesight_detector as fd_mod

    class _Stub(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.stem = nn.Conv2d(3, 64, 7, 4, 3)
            self.down1 = nn.Conv2d(64, 64, 3, 2, 1)
            self.down2 = nn.Conv2d(64, 64, 3, 2, 1)

        def forward(self, x):
            p3 = self.stem(x); p4 = self.down1(p3); p5 = self.down2(p4)
            return p3, p4, p5

    monkeypatch.setattr(fd_mod, "DINOv2Backbone", _Stub)
    model = build_model(_firesight_cfg())
    assert isinstance(model, FireSightDetector)


def test_build_model_unknown_type_raises():
    bad_cfg = _ccpe_cfg()
    bad_cfg["model"]["type"] = "definitely-not-a-model"
    with pytest.raises((ValueError, KeyError)):
        build_model(bad_cfg)
