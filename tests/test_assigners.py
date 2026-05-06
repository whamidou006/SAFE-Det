"""Tests for the unified label-assignment factory.

Runs the same minimal smoke-detection scenario through SimOTA, TAL, and
DSLA and verifies that each one:
  - returns the expected output shapes/dtypes,
  - produces at least one positive sample for an obviously-overlapping
    prediction/GT pair,
  - degenerates to all-negative output when given no GTs,
  - is constructable via the ``build_assigner`` factory.
"""

import pytest
import torch

from utils.assigner import (
    SimOTAAssigner,
    TALAssigner,
    DSLAAssigner,
    build_assigner,
)


def _toy_inputs(num_pred=64, img_size=128, device='cpu'):
    """Synthetic anchor grid at stride 16 over 128x128, plus one well-
    overlapping prediction and one obvious GT box."""
    side = img_size // 16
    yv, xv = torch.meshgrid(
        torch.arange(side) * 16 + 8,
        torch.arange(side) * 16 + 8,
        indexing='ij',
    )
    points = torch.stack([xv.flatten(), yv.flatten()], dim=1).float()
    strides = torch.full((points.shape[0],), 16.0)

    # Build pred boxes centred on every anchor with size 16.
    pred_boxes = torch.cat([points - 8, points + 8], dim=1).float()
    pred_scores = torch.full((pred_boxes.shape[0],), 0.6)

    # One GT box that overlaps a handful of anchors near the centre.
    gt_bboxes = torch.tensor([[56.0, 56.0, 88.0, 88.0]])
    gt_labels = torch.tensor([1], dtype=torch.long)

    return (pred_scores.to(device), pred_boxes.to(device),
            gt_bboxes.to(device), gt_labels.to(device),
            points.to(device), strides.to(device))


@pytest.mark.parametrize('name,cls', [
    ('simota', SimOTAAssigner),
    ('tal', TALAssigner),
    ('dsla', DSLAAssigner),
])
def test_assigner_factory_returns_expected_class(name, cls):
    a = build_assigner(name)
    assert isinstance(a, cls)


def test_assigner_factory_unknown_raises():
    with pytest.raises(ValueError):
        build_assigner('does_not_exist')


@pytest.mark.parametrize('name', ['simota', 'tal', 'dsla'])
def test_assigner_basic_assignment(name):
    pred_scores, pred_boxes, gt_bboxes, gt_labels, points, strides = _toy_inputs()
    a = build_assigner(name)
    labels, boxes, scores, fg = a.assign(
        pred_scores, pred_boxes, gt_bboxes, gt_labels, points, strides,
        num_classes=2,
    )
    n = pred_boxes.shape[0]
    assert labels.shape == (n,) and labels.dtype == torch.long
    assert boxes.shape == (n, 4)
    assert scores.shape == (n,)
    assert fg.shape == (n,) and fg.dtype == torch.bool
    # At least one positive must be selected for an obvious GT/pred match.
    assert fg.sum().item() > 0, f"{name}: produced no positive samples"
    # Every positive must carry the GT label.
    assert (labels[fg] == 1).all(), f"{name}: positive labels do not match GT"


@pytest.mark.parametrize('name', ['simota', 'tal', 'dsla'])
def test_assigner_no_gt_returns_all_negative(name):
    pred_scores, pred_boxes, _, _, points, strides = _toy_inputs()
    empty_gt = torch.zeros((0, 4))
    empty_lab = torch.zeros((0,), dtype=torch.long)
    a = build_assigner(name)
    labels, boxes, scores, fg = a.assign(
        pred_scores, pred_boxes, empty_gt, empty_lab, points, strides,
        num_classes=2,
    )
    assert fg.sum().item() == 0
    assert (labels == -1).all()


def test_tal_uses_alignment_metric_as_score():
    """For TAL the soft cls target should be in (0, 1] for positives."""
    pred_scores, pred_boxes, gt_bboxes, gt_labels, points, strides = _toy_inputs()
    a = TALAssigner(topk=8)
    _, _, scores, fg = a.assign(
        pred_scores, pred_boxes, gt_bboxes, gt_labels, points, strides,
        num_classes=2,
    )
    pos = scores[fg]
    assert (pos > 0).all() and (pos <= 1).all()


def test_dsla_dynamic_k_at_least_one():
    """DSLA's dynamic-k must always select at least one positive when the
    candidate set is non-empty, even if all IoUs are tiny."""
    pred_scores, pred_boxes, gt_bboxes, gt_labels, points, strides = _toy_inputs()
    # Make all preds slightly overlap the GT (small IoU) so dyn_k could
    # round to zero without the .clamp(min=1) safety.
    pred_boxes = pred_boxes + 0.1
    a = DSLAAssigner()
    _, _, _, fg = a.assign(
        pred_scores, pred_boxes, gt_bboxes, gt_labels, points, strides,
        num_classes=2,
    )
    assert fg.sum().item() >= 1
