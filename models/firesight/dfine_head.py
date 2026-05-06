"""
D-FINE / DEIM transformer-decoder head adapter for SAFE-Det.

D-FINE is the user's production detector and the natural successor to the
hand-rolled YOLOX head used elsewhere in SAFE-Det. Re-implementing it here
would duplicate ~3.5 kLOC; instead this module *imports* the existing
implementation from the user's local Condor-evaluation checkout.

To enable
---------

In the YAML config:

    model:
      type: firesight
      head_type: dfine                  # was: 'yolox'
      dfine_source: /path/to/Condor-evaluation     # optional; auto-detected
      dfine_kwargs: { ... }             # forwarded to DFINETransformer ctor

The adapter searches, in order:

1. ``cfg.model.dfine_source`` if set,
2. ``$CONDOR_EVALUATION_ROOT`` if set,
3. ``/home/whamidouche/ssdprivate/Condor-evaluation`` (default checkout).

If the import fails, a clear, actionable RuntimeError is raised
explaining where the source is expected and how to point at a different
checkout — *not* a silent fallback. Silent fallbacks were the bug we
removed from ``DINOv2Backbone`` earlier in this session.

Training caveat
---------------

The D-FINE decoder produces DETR-style outputs
``{'pred_logits': (B, Q, C), 'pred_boxes': (B, Q, 4)}`` and is trained
with Hungarian matching + focal cls + GIoU + L1 + FDR distillation,
all in ``rtv4_criterion.RTV4Criterion``. The current ``train.py`` loop
expects YOLOX-style outputs and uses ``YOLOXLoss``. Selecting
``head_type: dfine`` therefore requires a separate training entrypoint
that uses the matching D-FINE criterion (see TODO at the bottom of this
file for the integration point).
"""

import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


_DEFAULT_CONDOR_EVAL_ROOT = '/home/whamidouche/ssdprivate/Condor-evaluation'


def _resolve_rtv4_root(explicit: Optional[str] = None) -> str:
    """Locate the user's Condor-evaluation checkout that ships rtv4."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    if 'CONDOR_EVALUATION_ROOT' in os.environ:
        candidates.append(os.environ['CONDOR_EVALUATION_ROOT'])
    candidates.append(_DEFAULT_CONDOR_EVAL_ROOT)
    for root in candidates:
        decoder = os.path.join(root, 'engine', 'rtv4', 'dfine_decoder.py')
        if os.path.isfile(decoder):
            return root
    raise RuntimeError(
        "DFINEHeadAdapter could not locate the rtv4 source.\n"
        "Looked in:\n  - "
        + "\n  - ".join(candidates)
        + "\nProvide the correct path via cfg.model.dfine_source, set "
          "$CONDOR_EVALUATION_ROOT, or check out "
          "https://github.com/whamidou006/Condor-evaluation."
    )


def _import_dfine_transformer(source: Optional[str] = None):
    """Lazy import of ``engine.rtv4.dfine_decoder.DFINETransformer``."""
    root = _resolve_rtv4_root(source)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from engine.rtv4.dfine_decoder import DFINETransformer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "DFINEHeadAdapter located the rtv4 source at "
            f"{root!r} but failed to import "
            "engine.rtv4.dfine_decoder.DFINETransformer. "
            "Make sure the Condor-evaluation checkout is intact and that "
            "its dependencies (torch, transformers) are installed in the "
            f"current environment.\nOriginal error: {e}"
        ) from e
    return DFINETransformer


class DFINEHeadAdapter(nn.Module):
    """Thin wrapper around :class:`DFINETransformer` so that
    :class:`~models.firesight.firesight_detector.FireSightDetector` can
    swap it in via ``head_type: dfine``.

    The adapter exposes only the forward path. Decoding (post-processing
    boxes from queries) and training loss are handled by the D-FINE
    criterion, not by SAFE-Det's YOLOX-style train loop. See module
    docstring for the integration TODO.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels=(256, 256, 256),
                 source: Optional[str] = None,
                 dfine_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        DFINETransformer = _import_dfine_transformer(source)
        kwargs: Dict[str, Any] = dict(
            num_classes=num_classes,
            feat_channels=list(in_channels),
        )
        if dfine_kwargs:
            kwargs.update(dfine_kwargs)
        self.decoder = DFINETransformer(**kwargs)
        self.num_classes = num_classes

    def forward(self, features, targets=None):
        # The DFINETransformer expects a list of multi-scale feature maps
        # already projected to the encoder's hidden_dim. The neck produces
        # exactly that for SAFE-Det (PAFPN with uniform channels).
        # ``targets`` must be passed in training mode to enable the
        # contrastive denoising group (CDN); it is ignored at eval.
        return self.decoder(list(features), targets)

    # ------------------------------------------------------------------ #
    # Integration STATUS (handled in models/firesight/dfine_runtime.py): #
    # ------------------------------------------------------------------ #
    # 1. train.py: dispatches on FireSightDetector.head_type == 'dfine'  #
    #    and uses build_dfine_criterion() instead of YOLOXLoss.          #
    # 2. eval.py: dispatches on the same flag and uses                   #
    #    build_dfine_postprocessor() to decode queries -> per-image      #
    #    {labels, boxes, scores} dicts, which are then folded into the   #
    #    same per-class AP loop the YOLOX path uses.                     #
    # 3. Targets: SAFE-Det's dataset still emits (N, 5) flat tensors;    #
    #    convert_yolox_targets_to_dfine() converts them on the fly to    #
    #    list-of-dict cxcywh-normalised at criterion call time. No       #
    #    dataset changes required.                                        #
    # ------------------------------------------------------------------ #
