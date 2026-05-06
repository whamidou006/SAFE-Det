"""D-FINE training/eval runtime helpers.

Bridges the YOLOX-style train/eval loops in ``train.py`` / ``eval.py``
with the DETR-style criterion (``RTv4Criterion``) and post-processor
(``PostProcessor``) shipped in the user's local Condor-evaluation
checkout. Everything in this module is *lazy*: nothing is imported from
rtv4 until you actually call one of the builders, so SAFE-Det remains
runnable on machines that don't have the rtv4 source.

Three things are exposed:

- :func:`convert_yolox_targets_to_dfine` — adapt the dataset's
  ``(N, 5)`` ``[cls, x1, y1, w, h]`` pixel-space tensors to the
  ``[{'labels': (M,), 'boxes': (M, 4) cxcywh-normalised}]`` list-of-dict
  format that DETR criteria and matchers expect.
- :func:`build_dfine_criterion` — construct
  ``RTv4Criterion`` + ``HungarianMatcher`` with sensible defaults for
  smoke/fire (2 classes, no MAL/distill).
- :func:`build_dfine_postprocessor` — construct ``PostProcessor`` for
  inference-time decoding of query outputs into class-major detections.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .dfine_head import _import_dfine_transformer  # noqa: F401  (warm path-resolver)
from .dfine_head import _resolve_rtv4_root


__all__ = [
    'convert_yolox_targets_to_dfine',
    'build_dfine_criterion',
    'build_dfine_postprocessor',
]


# --------------------------------------------------------------------------- #
# Target conversion                                                           #
# --------------------------------------------------------------------------- #


def convert_yolox_targets_to_dfine(
        targets: List[torch.Tensor],
        img_size: int,
        device: Optional[torch.device] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Convert SAFE-Det's flat ``(N, 5)`` ``[cls, x1, y1, w, h]`` pixel
    targets into the ``[{'labels', 'boxes'}]`` DETR list-of-dict format.

    The DETR/D-FINE convention is **cxcywh** boxes normalised to
    ``[0, 1]`` by the input image size. SAFE-Det's dataset emits
    pixel-space xywh, so we offset by half the box, then divide by
    ``img_size``.
    """
    out: List[Dict[str, torch.Tensor]] = []
    for tgt in targets:
        if tgt.numel() == 0:
            out.append({
                'labels': torch.zeros((0,), dtype=torch.long,
                                      device=device or tgt.device),
                'boxes': torch.zeros((0, 4),
                                     device=device or tgt.device),
            })
            continue
        if device is not None:
            tgt = tgt.to(device)
        labels = tgt[:, 0].long()
        x1 = tgt[:, 1]
        y1 = tgt[:, 2]
        w = tgt[:, 3]
        h = tgt[:, 4]
        cx = (x1 + w * 0.5) / img_size
        cy = (y1 + h * 0.5) / img_size
        wn = w / img_size
        hn = h / img_size
        boxes = torch.stack([cx, cy, wn, hn], dim=1).clamp(0.0, 1.0)
        out.append({'labels': labels, 'boxes': boxes})
    return out


# --------------------------------------------------------------------------- #
# Criterion                                                                   #
# --------------------------------------------------------------------------- #


class _DFINECriterionWrapper(nn.Module):
    """Adapter around ``RTv4Criterion`` whose ``__call__`` matches the
    interface ``train.py``'s loop expects:

        total_loss, log_dict = wrapper(model_outputs, targets, img_size)

    ``model_outputs`` is the dict returned by
    :class:`DFINEHeadAdapter.forward`. ``targets`` is the *raw* YOLOX
    target list — conversion happens here so the train loop stays
    head-agnostic.
    """

    def __init__(self, criterion: nn.Module, num_classes: int):
        super().__init__()
        self.criterion = criterion
        self.num_classes = num_classes

    def __call__(self, model_outputs: Dict[str, torch.Tensor],
                 targets: List[torch.Tensor], img_size: int):
        # Match the device of the predicted boxes.
        device = model_outputs['pred_boxes'].device
        dfine_targets = convert_yolox_targets_to_dfine(
            targets, img_size=img_size, device=device,
        )
        loss_dict = self.criterion(model_outputs, dfine_targets)
        weight = self.criterion.weight_dict
        total = sum(loss_dict[k] * weight[k]
                    for k in loss_dict if k in weight)
        log_dict = {k: v.detach().item() for k, v in loss_dict.items()}
        log_dict['num_fg'] = int(sum(t['labels'].numel()
                                     for t in dfine_targets))
        # Keep the legacy keys train.py logs so the format string doesn't
        # need to change.
        log_dict.setdefault('loss_cls',
                            float(loss_dict.get('loss_focal',
                                                loss_dict.get('loss_vfl',
                                                              torch.tensor(0.0))).detach()))
        log_dict.setdefault('loss_bbox',
                            float(loss_dict.get('loss_bbox',
                                                torch.tensor(0.0)).detach()))
        log_dict.setdefault('loss_obj', 0.0)
        return total, log_dict


def build_dfine_criterion(num_classes: int = 2,
                          source: Optional[str] = None,
                          weight_dict: Optional[Dict[str, float]] = None,
                          losses: Optional[List[str]] = None,
                          reg_max: int = 32,
                          use_focal_loss: bool = True) -> _DFINECriterionWrapper:
    """Construct ``RTv4Criterion`` + ``HungarianMatcher`` lazily.

    Resolution order matches :func:`DFINEHeadAdapter`:

    1. Explicit ``source`` (cfg ``dfine_source``) or
       ``$CONDOR_EVALUATION_ROOT`` if set,
    2. otherwise the vendored copy in :mod:`models.firesight.rtv4`.

    The defaults reproduce the small-model RT-DETRv4 / D-FINE recipe
    (focal cls + L1 + GIoU, no MAL, no teacher distillation).
    """
    root = _resolve_rtv4_root(source)
    if root is None:
        from .rtv4 import HungarianMatcher, RTv4Criterion  # vendored
    else:
        import sys
        if root not in sys.path:
            sys.path.insert(0, root)
        from engine.rtv4.matcher import HungarianMatcher  # type: ignore
        from engine.rtv4.rtv4_criterion import RTv4Criterion  # type: ignore

    matcher_weights = {'cost_class': 2.0, 'cost_bbox': 5.0, 'cost_giou': 2.0}
    matcher = HungarianMatcher(weight_dict=matcher_weights,
                               use_focal_loss=use_focal_loss)
    weight_dict = weight_dict or {
        'loss_focal': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0,
    }
    losses = losses or ['focal', 'boxes']
    criterion = RTv4Criterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        num_classes=num_classes,
        reg_max=reg_max,
    )
    _patch_focal_target_dtype(criterion)
    return _DFINECriterionWrapper(criterion, num_classes=num_classes)


def _patch_focal_target_dtype(criterion) -> None:
    """torchvision >= 0.20 rejects integer ``targets`` in
    ``sigmoid_focal_loss`` (raises *result type Float can't be cast to
    the desired output type Long*). The upstream rtv4 code builds
    targets via ``F.one_hot(...)`` which yields ``int64`` and was only
    safe with older torchvision versions.

    Rather than patch Condor-evaluation, we monkey-patch the three
    ``loss_labels_*`` methods on the criterion instance with thin
    wrappers that coerce ``target`` to ``float`` before delegating to
    the original implementation. The behaviour is identical with older
    torchvision (it would have cast anyway) and unblocks the modern
    torchvision the user has installed.
    """
    import torch.nn.functional as F
    import torchvision

    original_focal_loss = torchvision.ops.sigmoid_focal_loss

    def _focal_loss_float_targets(inputs, targets, *args, **kwargs):
        if targets.dtype != inputs.dtype:
            targets = targets.to(inputs.dtype)
        return original_focal_loss(inputs, targets, *args, **kwargs)

    # Replace the symbol on the criterion's *imported* module so we
    # don't globally monkey-patch torchvision for the rest of the
    # process — only RTv4Criterion's calls re-bind through this name.
    import sys
    crit_module = sys.modules[criterion.__class__.__module__]
    crit_module.torchvision.ops.sigmoid_focal_loss = _focal_loss_float_targets


# --------------------------------------------------------------------------- #
# Post-processor                                                              #
# --------------------------------------------------------------------------- #


def build_dfine_postprocessor(num_classes: int = 2,
                              source: Optional[str] = None,
                              num_top_queries: int = 300,
                              use_focal_loss: bool = True):
    """Construct ``PostProcessor`` lazily.

    Returns a callable with the same signature as the upstream class:

        results = postprocessor(model_outputs, orig_target_sizes)

    ``orig_target_sizes`` is a ``(B, 2)`` tensor of ``(H, W)`` pixels.
    Each entry of ``results`` is ``{'labels', 'boxes', 'scores'}``.

    Resolution order matches :func:`build_dfine_criterion`.
    """
    root = _resolve_rtv4_root(source)
    if root is None:
        from .rtv4 import PostProcessor  # vendored
    else:
        import sys
        if root not in sys.path:
            sys.path.insert(0, root)
        from engine.rtv4.postprocessor import PostProcessor  # type: ignore
    return PostProcessor(num_classes=num_classes,
                         use_focal_loss=use_focal_loss,
                         num_top_queries=num_top_queries)
