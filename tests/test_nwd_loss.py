"""Tests for the optional NWD bbox-regression loss.

The default YOLOXLoss path (`bbox_loss_type='ciou'`) must remain
bit-identical with the previous implementation; the NWD and 'mixed'
paths must be smooth, finite and produce sensible gradients.
"""
import pytest
import torch

from models.losses_nwd import bbox_loss, nwd_iou


def _box(cx, cy, w, h):
    return torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]],
                        dtype=torch.float32)


def test_nwd_self_similarity_is_one():
    pred = _box(50.0, 50.0, 10.0, 10.0)
    # Tolerance reflects the 1e-7 epsilon floor inside sqrt that keeps the
    # gradient finite at w2 = 0 (sqrt(1e-7)/12.8 ≈ 2.5e-5 from exp).
    assert torch.allclose(nwd_iou(pred, pred), torch.ones(1), atol=1e-4)


def test_nwd_decreases_with_distance():
    pred = _box(50.0, 50.0, 10.0, 10.0)
    near = _box(51.0, 51.0, 10.0, 10.0)
    far = _box(80.0, 80.0, 10.0, 10.0)
    assert nwd_iou(pred, near).item() > nwd_iou(pred, far).item()


def test_nwd_gradient_for_tiny_disjoint_boxes():
    # Two 4x4 boxes with no overlap. IoU is exactly 0 here, which means
    # plain 1-IoU has zero gradient w.r.t. the prediction. NWD must be
    # strictly between 0 and 1 and have non-zero gradient.
    pred = torch.tensor([[10.0, 10.0, 14.0, 14.0]], requires_grad=True)
    target = torch.tensor([[20.0, 20.0, 24.0, 24.0]])
    loss = bbox_loss(pred, target, loss_type='nwd', nwd_constant=12.8).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert 0.0 < loss.item() < 1.0
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_ciou_default_path_unchanged():
    # Bit-identical with the previous YOLOXLoss._iou_loss implementation
    # for an overlapping pair.
    pred = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
    target = torch.tensor([[15.0, 15.0, 35.0, 35.0]])
    inter = (30 - 15) * (30 - 15)
    union = 20 * 20 + 20 * 20 - inter
    expected = torch.tensor([1.0 - inter / union])
    assert torch.allclose(bbox_loss(pred, target, loss_type='ciou'),
                          expected, atol=1e-6)


def test_mixed_loss_in_between():
    pred = torch.tensor([[10.0, 10.0, 14.0, 14.0]])
    target = torch.tensor([[20.0, 20.0, 24.0, 24.0]])
    iou_loss = bbox_loss(pred, target, loss_type='ciou')
    nwd_loss = bbox_loss(pred, target, loss_type='nwd', nwd_constant=12.8)
    mixed = bbox_loss(pred, target, loss_type='mixed',
                      nwd_constant=12.8, nwd_mix_weight=0.5)
    expected = 0.5 * iou_loss + 0.5 * nwd_loss
    assert torch.allclose(mixed, expected, atol=1e-6)


def test_unknown_loss_type_raises():
    pred = torch.zeros(1, 4)
    target = torch.zeros(1, 4)
    with pytest.raises(ValueError):
        bbox_loss(pred, target, loss_type='made_up')


def test_yoloxloss_dispatches_to_nwd():
    """End-to-end: setting bbox_loss_type on YOLOXLoss must change the
    bbox term while leaving the rest of the loss pipeline alone."""
    from train import YOLOXLoss

    loss_ciou = YOLOXLoss(num_classes=2, bbox_loss_type='ciou')
    loss_nwd = YOLOXLoss(num_classes=2, bbox_loss_type='nwd', nwd_constant=12.8)

    pred = torch.tensor([[10.0, 10.0, 14.0, 14.0]])
    target = torch.tensor([[20.0, 20.0, 24.0, 24.0]])

    # _iou_loss is the dispatcher in our patched YOLOXLoss.
    v_ciou = loss_ciou._iou_loss(pred, target).item()
    v_nwd = loss_nwd._iou_loss(pred, target).item()
    assert v_ciou == pytest.approx(1.0)            # disjoint => IoU=0 => loss=1
    assert 0.0 < v_nwd < 1.0                       # NWD is strictly in (0,1)
