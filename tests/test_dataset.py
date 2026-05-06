"""Tests for the FireSmokeDataset (YOLO-format loader)."""
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from utils.dataset import FireSmokeDataset, collate_fn


@pytest.fixture
def tiny_yolo_dataset(tmp_path):
    """Build a 4-image YOLO-format dataset with smoke and fire annotations."""
    img_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    img_dir.mkdir()
    label_dir.mkdir()

    rng = np.random.default_rng(0)
    for i in range(4):
        # Write an image
        img = (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
        cv2.imwrite(str(img_dir / f"img_{i:03d}.jpg"), img)
        # Write a YOLO label (1 smoke + 1 fire per image, except #3 = empty)
        if i < 3:
            with open(label_dir / f"img_{i:03d}.txt", "w") as f:
                # class cx cy w h (normalized)
                f.write("0 0.5 0.5 0.2 0.2\n")
                f.write("1 0.3 0.3 0.1 0.1\n")
        else:
            (label_dir / f"img_{i:03d}.txt").write_text("")

    return img_dir, label_dir


def test_dataset_basic_load(tiny_yolo_dataset):
    img_dir, label_dir = tiny_yolo_dataset
    ds = FireSmokeDataset(
        img_dir=str(img_dir), label_dir=str(label_dir),
        img_size=128, augment=False,
    )
    assert len(ds) == 4
    img, targets, idx = ds[0]
    assert img.shape == (3, 128, 128)
    assert img.dtype == torch.float32
    assert (img >= 0).all() and (img <= 1).all()
    assert targets.shape == (2, 5)         # 2 boxes, [cls, x1, y1, w, h]
    assert targets[0, 0].item() == 0       # smoke class
    assert targets[1, 0].item() == 1       # fire class


def test_dataset_empty_image(tiny_yolo_dataset):
    img_dir, label_dir = tiny_yolo_dataset
    ds = FireSmokeDataset(img_dir=str(img_dir), label_dir=str(label_dir),
                          img_size=128, augment=False)
    img, targets, _ = ds[3]                # img_003 is empty
    assert targets.shape == (0, 5)
    assert img.shape == (3, 128, 128)


def test_dataset_with_augmentation_does_not_crash(tiny_yolo_dataset):
    img_dir, label_dir = tiny_yolo_dataset
    ds = FireSmokeDataset(img_dir=str(img_dir), label_dir=str(label_dir),
                          img_size=128, augment=True,
                          mosaic_prob=1.0, mixup_prob=0.5)
    for i in range(len(ds)):
        img, targets, _ = ds[i]
        assert img.shape == (3, 128, 128)
        assert targets.shape[1] == 5
        # Targets must lie within image bounds [0, 128]
        if len(targets) > 0:
            assert (targets[:, 1:].min() >= 0).item()
            assert (targets[:, 1] + targets[:, 3]).max().item() <= 128 + 1e-3
            assert (targets[:, 2] + targets[:, 4]).max().item() <= 128 + 1e-3


def test_collate_variable_targets(tiny_yolo_dataset):
    img_dir, label_dir = tiny_yolo_dataset
    ds = FireSmokeDataset(img_dir=str(img_dir), label_dir=str(label_dir),
                          img_size=128, augment=False)
    batch = [ds[i] for i in range(4)]
    imgs, targets, indices = collate_fn(batch)
    assert imgs.shape == (4, 3, 128, 128)
    assert isinstance(targets, list) and len(targets) == 4
    # Image #3 is empty
    assert targets[3].shape == (0, 5)
