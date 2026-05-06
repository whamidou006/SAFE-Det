"""Pytest fixtures shared across the SAFE-Det test suite.

We add the repo root to sys.path so tests can import `models`, `utils`, etc.
without installing the project. Heavy components (DINOv2 / CUDA) are kept
optional behind fixtures so the suite runs CPU-only by default.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Force offline mode for DINOv2 fetches in tests so we never hit the network.
os.environ.setdefault("SAFE_DET_OFFLINE_BACKBONE", "1")
# Disable HF/torch hub network access during tests.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def device() -> torch.device:
    """Device for tests — CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_image(device) -> torch.Tensor:
    """Tiny 4-batch 128×128 RGB image for fast forward-pass tests."""
    torch.manual_seed(0)
    return torch.randn(2, 3, 128, 128, device=device)


@pytest.fixture
def small_features(device) -> torch.Tensor:
    """Tiny 32-channel 16×16 feature map for module unit tests."""
    torch.manual_seed(0)
    return torch.randn(2, 32, 16, 16, device=device)
