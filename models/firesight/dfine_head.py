"""
D-FINE / DEIM transformer-decoder head adapter for SAFE-Det.

D-FINE is the user's production detector and the natural successor to the
hand-rolled YOLOX head used elsewhere in SAFE-Det. The implementation
itself is vendored under :mod:`models.firesight.rtv4` (see that package's
``__init__.py`` for provenance / license notes), so this module is a
thin wrapper that constructs the transformer with SAFE-Det defaults and
exposes a forward method matching FireSightDetector's expectations.

To enable
---------

In the YAML config:

    model:
      type: firesight
      head_type: dfine                  # was: 'yolox'
      dfine_source: /optional/external  # OPTIONAL — see below
      dfine_kwargs: { ... }             # forwarded to DFINETransformer ctor

By default the head loads from the vendored
``models.firesight.rtv4`` package and *no external checkout is
required*. The ``dfine_source`` option remains as an escape hatch:
when set, it loads ``engine.rtv4.dfine_decoder.DFINETransformer`` from
that path instead. This is useful when validating against an upstream
Condor-evaluation revision without re-vendoring.
"""

import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


_DEFAULT_CONDOR_EVAL_ROOT = '/home/whamidouche/ssdprivate/Condor-evaluation'


def _resolve_rtv4_root(explicit: Optional[str] = None) -> Optional[str]:
    """Locate an external Condor-evaluation checkout that ships rtv4.

    Returns ``None`` when no explicit override is requested — the caller
    should then fall back to the vendored copy bundled in
    :mod:`models.firesight.rtv4`. Returns a string path when an explicit
    override is provided (cfg ``dfine_source`` or
    ``$CONDOR_EVALUATION_ROOT``) *and* the path actually exists; raises
    :class:`RuntimeError` if an explicit override is provided but the
    checkout is missing (silent fallback would hide config errors).
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    if 'CONDOR_EVALUATION_ROOT' in os.environ:
        candidates.append(os.environ['CONDOR_EVALUATION_ROOT'])

    if not candidates:
        return None  # no override → use vendored copy

    for root in candidates:
        decoder = os.path.join(root, 'engine', 'rtv4', 'dfine_decoder.py')
        if os.path.isfile(decoder):
            return root

    raise RuntimeError(
        "DFINEHeadAdapter was asked to load D-FINE from an external "
        "Condor-evaluation checkout but the path is not valid. "
        "Looked in:\n  - " + "\n  - ".join(candidates)
        + "\nEither remove the override (and the vendored copy in "
          "models.firesight.rtv4 will be used) or set "
          "cfg.model.dfine_source / $CONDOR_EVALUATION_ROOT to a "
          "directory containing engine/rtv4/dfine_decoder.py."
    )


def _import_dfine_transformer(source: Optional[str] = None):
    """Resolve and return the ``DFINETransformer`` class.

    Priority:

    1. If ``source`` (cfg ``dfine_source``) or ``$CONDOR_EVALUATION_ROOT``
       is set, load from that external checkout.
    2. Otherwise load from the vendored ``models.firesight.rtv4``
       package — the default and only requirement for an out-of-the-box
       D-FINE training run inside SAFE-Det.
    """
    root = _resolve_rtv4_root(source)
    if root is None:
        from .rtv4 import DFINETransformer  # vendored
        return DFINETransformer

    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from engine.rtv4.dfine_decoder import DFINETransformer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "DFINEHeadAdapter located an external rtv4 source at "
            f"{root!r} but failed to import "
            "engine.rtv4.dfine_decoder.DFINETransformer. "
            "Make sure the Condor-evaluation checkout is intact and "
            "its dependencies are installed in the current environment, "
            "or remove the override to use the vendored copy.\n"
            f"Original error: {e}"
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
