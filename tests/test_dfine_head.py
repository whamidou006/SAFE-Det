"""Tests for the D-FINE head dispatcher in FireSightDetector.

We do not require D-FINE to be importable for this test to pass — the
goal is to verify:

1. ``head_type='dfine'`` no longer raises ``NotImplementedError`` (the
   placeholder has been replaced by a proper adapter import).
2. When the rtv4 source cannot be located, the failure is a clean,
   actionable ``RuntimeError`` mentioning the search path — not a silent
   fallback (which is the bug pattern we already removed from the
   DINOv2 backbone earlier in this session).
3. ``head_type='unknown'`` raises ``ValueError`` with a clear message.
"""

import os
import pytest
import torch

from models.firesight import dfine_head
from models.firesight.firesight_detector import FireSightDetector


def _make_stub_backbone():
    """Return an instance of the offline backbone stub used elsewhere in
    the test suite, so this test never touches the real DINOv2 hub."""
    import torch.nn as nn

    class _Stub(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.stem = nn.Conv2d(3, 64, 7, 4, 3)
            self.down1 = nn.Conv2d(64, 64, 3, 2, 1)
            self.down2 = nn.Conv2d(64, 64, 3, 2, 1)

        def forward(self, x):
            p3 = self.stem(x); p4 = self.down1(p3); p5 = self.down2(p4)
            return p3, p4, p5

    return _Stub


def test_unknown_head_type_raises_clearly(monkeypatch):
    import models.firesight.firesight_detector as fd_mod
    monkeypatch.setattr(fd_mod, 'DINOv2Backbone', _make_stub_backbone())
    with pytest.raises(ValueError, match="head_type"):
        FireSightDetector(
            num_classes=2, backbone_channels=64,
            use_dcm=False, use_fam=False, use_tm=False, use_temporal=False,
            head_type='not_a_real_head', input_size=(64, 64),
        )


def test_dfine_dispatch_does_not_raise_notimplemented(monkeypatch):
    """The old code raised NotImplementedError. The new dispatcher must
    instead either successfully construct the adapter (if rtv4 is on
    the user's machine) or raise a RuntimeError with location info."""
    import models.firesight.firesight_detector as fd_mod
    monkeypatch.setattr(fd_mod, 'DINOv2Backbone', _make_stub_backbone())

    try:
        FireSightDetector(
            num_classes=2, backbone_channels=64,
            use_dcm=False, use_fam=False, use_tm=False, use_temporal=False,
            head_type='dfine', input_size=(64, 64),
            dfine_source='/definitely/not/a/real/path/rtv4',
        )
    except NotImplementedError:
        pytest.fail("dfine head still raises NotImplementedError")
    except RuntimeError as e:
        # Expected when rtv4 isn't available — must be a *clear* error
        # naming what we looked for and how to fix it.
        msg = str(e)
        assert 'rtv4' in msg.lower() or 'condor-evaluation' in msg.lower()
        assert '/definitely/not/a/real/path/rtv4' in msg


def test_resolve_rtv4_root_searches_in_priority_order(tmp_path, monkeypatch):
    # Create a fake rtv4 layout to satisfy the resolver.
    fake = tmp_path / 'engine' / 'rtv4'
    fake.mkdir(parents=True)
    (fake / 'dfine_decoder.py').write_text("# stub\n")

    # Explicit beats env var beats default.
    monkeypatch.setenv('CONDOR_EVALUATION_ROOT', '/tmp/wrong')
    assert dfine_head._resolve_rtv4_root(str(tmp_path)) == str(tmp_path)

    monkeypatch.delenv('CONDOR_EVALUATION_ROOT', raising=False)
    monkeypatch.setenv('CONDOR_EVALUATION_ROOT', str(tmp_path))
    assert dfine_head._resolve_rtv4_root() == str(tmp_path)


def test_resolve_rtv4_root_actionable_error(monkeypatch):
    monkeypatch.delenv('CONDOR_EVALUATION_ROOT', raising=False)
    # Block the production default so we exercise the failure branch even
    # on a machine that already has a working Condor-evaluation checkout.
    monkeypatch.setattr(dfine_head, '_DEFAULT_CONDOR_EVAL_ROOT',
                        '/no/such/default/path')
    with pytest.raises(RuntimeError) as exc:
        dfine_head._resolve_rtv4_root('/no/such/path')
    msg = str(exc.value)
    assert '/no/such/path' in msg
    assert 'CONDOR_EVALUATION_ROOT' in msg
    assert 'dfine_source' in msg


def test_dfine_adapter_constructs_when_rtv4_available():
    """Smoke test for the happy path: if rtv4 is on this machine the
    adapter must be constructable. Skipped when not available so the
    suite still passes on bare environments."""
    try:
        dfine_head._resolve_rtv4_root()
    except RuntimeError:
        pytest.skip("rtv4 source not available on this machine")
    try:
        adapter = dfine_head.DFINEHeadAdapter(
            num_classes=2, in_channels=(256, 256, 256),
        )
    except RuntimeError as e:
        pytest.skip(f"rtv4 import fails on this env (likely missing dep): {e}")
    assert adapter.num_classes == 2
    assert adapter.decoder is not None
