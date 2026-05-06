"""
Vendored subset of the rtv4 (RT-DETRv4 / D-FINE) implementation from the
user's Condor-evaluation repository, sufficient to run a D-FINE detector
inside SAFE-Det without an external dependency on Condor-evaluation.

Origin (last sync 2026-05-06):
    https://github.com/whamidou006/Condor-evaluation/tree/main/engine/rtv4

Vendored files preserve their original copyright headers. The upstream
implementation is licensed under Apache-2.0 (DETR © Facebook,
D-FINE © Peng et al., DEIM © DEIM authors).

Shims provided here (replacing what would normally come from the parent
``engine.core`` and ``engine.misc.dist_utils`` packages — neither of
which is needed at inference / training time inside SAFE-Det):

* :func:`register` — no-op replacement for the DI-registry decorator
  used throughout the upstream code. SAFE-Det does not use the registry,
  so the decorator is a pass-through.

* :func:`is_dist_available_and_initialized`,
  :func:`get_world_size` — minimal copies of the two helpers from
  ``engine.misc.dist_utils`` that ``rtv4_criterion`` calls. Behaviour is
  identical to the upstream implementation.

Public API re-exported here (these are the only symbols SAFE-Det imports):

* :class:`DFINETransformer` — the D-FINE decoder head
* :class:`HungarianMatcher` — DETR Hungarian assignment
* :class:`RTv4Criterion` — DETR-style multi-scale criterion
* :class:`PostProcessor` — query → boxes/labels/scores at eval time

Modifications relative to upstream:

* ``from ..core import register`` →
  ``from . import register`` (local no-op shim)
* ``from ..misc.dist_utils import ...`` →
  ``from . import get_world_size, is_dist_available_and_initialized``
"""

from typing import Any, Callable

import torch


# --------------------------------------------------------------------------- #
# Shim 1: ``register`` decorator from upstream ``engine.core``                 #
# --------------------------------------------------------------------------- #


def register(*args: Any, **kwargs: Any) -> Callable:
    """No-op stand-in for ``engine.core.register``.

    Upstream uses this decorator to register classes into a global
    dependency-injection registry that the YAML-config loader consults
    at runtime. SAFE-Det builds rtv4 components directly via
    ``models/firesight/dfine_runtime.py`` so the registry is never
    queried; the decorator therefore needs to do nothing.

    Supports both ``@register`` and ``@register()`` call styles.
    """
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # Used as ``@register`` (no parens) — args[0] is the class.
        return args[0]

    def _decorator(cls):
        return cls

    return _decorator


# --------------------------------------------------------------------------- #
# Shim 2: distributed helpers from upstream ``engine.misc.dist_utils``         #
# --------------------------------------------------------------------------- #


def is_dist_available_and_initialized() -> bool:
    """Return ``True`` only when ``torch.distributed`` is both available
    in this build *and* initialised (i.e. a process group exists)."""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Return the number of ranks in the current process group; ``1``
    when training non-distributed."""
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

# Lazy: only import the heavy modules on first access. Import order matters
# (criterion depends on matcher, decoder depends on utils + denoising etc.),
# but these top-level imports trigger the full chain in the right order.

from .matcher import HungarianMatcher  # noqa: E402,F401
from .postprocessor import PostProcessor  # noqa: E402,F401
from .rtv4_criterion import RTv4Criterion  # noqa: E402,F401
from .dfine_decoder import DFINETransformer  # noqa: E402,F401


__all__ = [
    'DFINETransformer',
    'HungarianMatcher',
    'PostProcessor',
    'RTv4Criterion',
    # Shims (re-exported for tests / internal use)
    'register',
    'get_world_size',
    'is_dist_available_and_initialized',
]
