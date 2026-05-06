"""Tests for SimOTA assigner and the YOLOXLoss training step."""
import sys
from pathlib import Path

import pytest
import torch

from utils.assigner import SimOTAAssigner

# YOLOXLoss lives in train.py (not a module). Import it directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import YOLOXLoss  # noqa: E402

from models.detector import CCPE_Detector


def test_simota_no_targets(device):
    asg = SimOTAAssigner()
    pred_scores = torch.rand(100, device=device)
    pred_boxes = torch.rand(100, 4, device=device) * 64
    gt_boxes = torch.zeros(0, 4, device=device)
    gt_labels = torch.zeros(0, dtype=torch.long, device=device)
    points = torch.rand(100, 2, device=device) * 64
    strides = torch.full((100,), 8.0, device=device)
    labels, bboxes, scores, fg = asg.assign(
        pred_scores, pred_boxes, gt_boxes, gt_labels, points, strides, num_classes=2)
    assert labels.shape == (100,)
    assert (labels == -1).all()
    assert fg.sum() == 0


def test_simota_with_targets(device):
    asg = SimOTAAssigner(center_radius=2.5, candidate_topk=10)
    torch.manual_seed(0)
    # Make 100 anchors arranged on an 8-px grid in a 80×80 image
    coords = torch.linspace(4, 76, 10, device=device)
    yv, xv = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack([xv.flatten(), yv.flatten()], 1)
    strides = torch.full((100,), 8.0, device=device)
    pred_boxes = torch.cat([points - 4, points + 4], 1)  # 8×8 boxes around each
    pred_scores = torch.full((100,), 0.5, device=device)
    # One GT box covering the centre of the image
    gt_boxes = torch.tensor([[30., 30., 50., 50.]], device=device)
    gt_labels = torch.tensor([0], device=device, dtype=torch.long)
    labels, bboxes, scores, fg = asg.assign(
        pred_scores, pred_boxes, gt_boxes, gt_labels, points, strides, num_classes=2)
    assert fg.sum() > 0
    assert (labels[fg] == 0).all()


def test_yolox_loss_with_targets(device):
    if device.type != "cuda":
        pytest.skip("YOLOXLoss expects CUDA tensors (uses .cuda() in train.py)")
    model = CCPE_Detector(
        num_classes=2, embed_dims=32, depths=(2, 2, 2, 2),
        num_heads=(2, 4, 8, 8), window_size=4, fpn_channels=32,
        input_size=(128, 128), contrast_steps=[1, 2, 4],
    ).to(device).train()
    loss_fn = YOLOXLoss(num_classes=2, img_size=128)

    x = torch.randn(2, 3, 128, 128, device=device)
    cls, bbox, obj = model(x)

    # Target format: [cls, x1, y1, w, h]
    targets = [
        torch.tensor([[0, 30., 30., 20., 20.],
                      [1, 60., 60., 10., 10.]], device=device),
        torch.zeros(0, 5, device=device),  # empty
    ]
    loss, loss_dict = loss_fn(cls, bbox, obj, targets, model.head)
    assert torch.isfinite(loss)
    assert loss > 0
    assert "loss_cls" in loss_dict and "loss_bbox" in loss_dict
    assert "loss_obj" in loss_dict and "num_fg" in loss_dict
    # Backward must work
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0


def test_yolox_loss_all_empty_targets(device):
    if device.type != "cuda":
        pytest.skip("YOLOXLoss expects CUDA tensors")
    model = CCPE_Detector(num_classes=2, embed_dims=32,
                         depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 8),
                         window_size=4, fpn_channels=32,
                         input_size=(128, 128), contrast_steps=[1, 2, 4]
                         ).to(device).train()
    loss_fn = YOLOXLoss(num_classes=2, img_size=128)
    x = torch.randn(2, 3, 128, 128, device=device)
    cls, bbox, obj = model(x)
    targets = [torch.zeros(0, 5, device=device), torch.zeros(0, 5, device=device)]
    loss, _ = loss_fn(cls, bbox, obj, targets, model.head)
    assert torch.isfinite(loss)
