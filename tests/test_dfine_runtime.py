"""Tests for D-FINE training/eval runtime wiring (target conversion +
criterion/postprocessor builders + train.py / eval.py dispatch).

Unit-level tests (target conversion + dispatch unit tests) always run.
End-to-end criterion / postprocessor tests are skipped automatically
when the rtv4 source is not importable in the current environment.
"""

import importlib
import os
import sys

import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.firesight.dfine_runtime import (  # noqa: E402
    convert_yolox_targets_to_dfine,
)


# --------------------------------------------------------------------------- #
# Target conversion                                                            #
# --------------------------------------------------------------------------- #


def test_convert_targets_centers_and_normalises():
    targets = [torch.tensor([[0.0, 100.0, 200.0, 80.0, 60.0],
                             [1.0, 0.0, 0.0, 1024.0, 1024.0]])]
    out = convert_yolox_targets_to_dfine(targets, img_size=1024)
    assert len(out) == 1
    labels = out[0]['labels']
    boxes = out[0]['boxes']
    assert labels.dtype == torch.long
    assert labels.tolist() == [0, 1]
    # First box: center = (100+40, 200+30) = (140, 230); normalized.
    assert torch.allclose(
        boxes[0],
        torch.tensor([140 / 1024, 230 / 1024, 80 / 1024, 60 / 1024]),
        atol=1e-6,
    )
    # Second box: full image, cxcywh = (0.5, 0.5, 1.0, 1.0).
    assert torch.allclose(
        boxes[1],
        torch.tensor([0.5, 0.5, 1.0, 1.0]),
        atol=1e-6,
    )


def test_convert_targets_handles_empty_batch():
    targets = [torch.zeros((0, 5)), torch.tensor([[1.0, 0.0, 0.0, 10.0, 20.0]])]
    out = convert_yolox_targets_to_dfine(targets, img_size=512)
    assert out[0]['labels'].numel() == 0
    assert out[0]['boxes'].shape == (0, 4)
    assert out[1]['labels'].tolist() == [1]


def test_convert_targets_clamps_to_unit_box():
    # Slightly out-of-frame box (e.g. mosaic edge) should clamp to [0,1].
    targets = [torch.tensor([[0.0, -10.0, -10.0, 1100.0, 1100.0]])]
    out = convert_yolox_targets_to_dfine(targets, img_size=1024)
    assert (out[0]['boxes'] >= 0.0).all()
    assert (out[0]['boxes'] <= 1.0).all()


def test_convert_targets_respects_device():
    targets = [torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])]
    out = convert_yolox_targets_to_dfine(
        targets, img_size=10, device=torch.device('cpu'),
    )
    assert out[0]['boxes'].device.type == 'cpu'


# --------------------------------------------------------------------------- #
# Criterion / postprocessor / end-to-end (skip if rtv4 not present)            #
# --------------------------------------------------------------------------- #


def _rtv4_available():
    try:
        from models.firesight.dfine_head import _resolve_rtv4_root
        _resolve_rtv4_root()
        return True
    except Exception:
        return False


pytestmark_rtv4 = pytest.mark.skipif(
    not _rtv4_available(),
    reason="rtv4 source not available — skipping D-FINE runtime tests",
)


@pytestmark_rtv4
def test_build_dfine_criterion_returns_callable_wrapper():
    from models.firesight.dfine_runtime import build_dfine_criterion
    crit = build_dfine_criterion(num_classes=2)
    # Synthetic D-FINE outputs: 1 image, 100 queries, 2 classes.
    # RTv4Criterion requires deep-supervision keys (`aux_outputs`,
    # `enc_aux_outputs`, `enc_meta`), so we fabricate them with the
    # same shape as the main outputs — matching what the real
    # DFINETransformer emits in training mode.
    B, Q, C = 1, 100, 2
    main = {
        'pred_logits': torch.zeros(B, Q, C),
        'pred_boxes': torch.full((B, Q, 4), 0.5),
    }
    outputs = dict(main)
    outputs['aux_outputs'] = [dict(main)]
    outputs['enc_aux_outputs'] = [dict(main)]
    outputs['enc_meta'] = {'class_agnostic': False}
    targets = [torch.tensor([[0.0, 100.0, 100.0, 50.0, 50.0]])]
    total, log = crit(outputs, targets, img_size=512)
    assert torch.is_tensor(total)
    assert torch.isfinite(total)
    assert 'loss_focal' in log or 'loss_vfl' in log
    assert log['num_fg'] == 1


@pytestmark_rtv4
def test_build_dfine_postprocessor_returns_per_image_dicts():
    from models.firesight.dfine_runtime import build_dfine_postprocessor
    pp = build_dfine_postprocessor(num_classes=2, num_top_queries=10)
    B, Q, C = 2, 100, 2
    outputs = {
        'pred_logits': torch.randn(B, Q, C),
        'pred_boxes': torch.full((B, Q, 4), 0.5),
    }
    sizes = torch.tensor([[512, 512], [1024, 1024]])
    out = pp(outputs, sizes)
    assert isinstance(out, list) and len(out) == B
    for r in out:
        assert set(r.keys()) >= {'labels', 'boxes', 'scores'}
        assert r['boxes'].ndim == 2 and r['boxes'].shape[1] == 4


# --------------------------------------------------------------------------- #
# Dispatch tests — train.py / eval.py branch correctly on head_type            #
# --------------------------------------------------------------------------- #


class _FakeYoloxHead(nn.Module):
    head_type = 'yolox'

    def forward(self, x):
        # Match YOLOX shape: (cls, bbox, obj) per scale, but the test only
        # cares that train_one_epoch picks the YOLOX branch.
        B = x.shape[0]
        return ([torch.zeros(B, 2, 8, 8, requires_grad=True)],
                [torch.zeros(B, 4, 8, 8, requires_grad=True)],
                [torch.zeros(B, 1, 8, 8, requires_grad=True)])


class _FakeDfineModel(nn.Module):
    head_type = 'dfine'

    def __init__(self, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(3, num_classes)
        self.num_classes = num_classes

    def forward(self, x, targets=None):
        # Return DETR-style outputs that are differentiable wrt self.proj
        # and live on the same device as the inputs (matches a real
        # DFINETransformer in training mode). Includes the deep-
        # supervision keys the criterion expects.
        B = x.shape[0]
        Q = 8
        feat = x.mean(dim=(2, 3))            # (B, 3)
        logits = self.proj(feat).unsqueeze(1).expand(B, Q, self.num_classes)
        boxes = torch.full((B, Q, 4), 0.5, device=x.device)
        main = {'pred_logits': logits, 'pred_boxes': boxes}
        out = dict(main)
        out['aux_outputs'] = [dict(main)]
        out['enc_aux_outputs'] = [dict(main)]
        out['enc_meta'] = {'class_agnostic': False}
        return out


def test_train_one_epoch_uses_dfine_branch_when_available():
    """Smoke-test the dispatch path inside train_one_epoch; runs only when
    the D-FINE criterion can actually be constructed (rtv4 available).
    Also requires CUDA because train_one_epoch hard-codes
    ``imgs.cuda(non_blocking=True)``.
    """
    if not _rtv4_available():
        pytest.skip("rtv4 source not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from train import train_one_epoch
    from models.firesight.dfine_runtime import build_dfine_criterion

    model = _FakeDfineModel(num_classes=2).cuda()
    crit = build_dfine_criterion(num_classes=2)

    img = torch.randn(1, 3, 32, 32)
    tgt = torch.tensor([[0.0, 5.0, 5.0, 10.0, 10.0]])

    class _OneBatchLoader:
        def __iter__(self_inner):
            yield img, [tgt], [0]

        def __len__(self_inner):
            return 1

    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    cfg = {'log_interval': 10,
           'model': {'type': 'firesight', 'num_classes': 2,
                     'input_size': [32, 32]}}
    train_one_epoch(model, _OneBatchLoader(), optim, scaler,
                    crit, epoch=0, rank=0, cfg=cfg)
