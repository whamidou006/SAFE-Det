"""Unit tests for Swin+CCPE backbone, PAFPN neck, and YOLOX head."""
import pytest
import torch

from models.swin_ccpe import SwinTransformerCCPE
from models.neck import YOLOXPAFPN
from models.head import YOLOXHeadSNSM
from models.detector import CCPE_Detector


@pytest.fixture
def tiny_swin(device):
    """A very small Swin-CCPE backbone for fast tests."""
    return SwinTransformerCCPE(
        in_channels=3,
        embed_dims=32,
        depths=(2, 2, 2, 2),
        num_heads=(2, 4, 8, 8),
        window_size=4,
        out_indices=(1, 2, 3),
        contrast_steps=[1, 2, 4],
    ).to(device)


def test_swin_backbone_three_scales(tiny_swin, device):
    x = torch.randn(1, 3, 128, 128, device=device)
    feats = tiny_swin(x)
    assert len(feats) == 3
    # Strides should be approximately 8, 16, 32 from a base patch size of 4
    # plus stage downsamplings of 2, 4, 8 respectively.
    expected_sizes = [(16, 16), (8, 8), (4, 4)]
    expected_chs = [32 * 2, 32 * 4, 32 * 8]
    for f, hw, c in zip(feats, expected_sizes, expected_chs):
        assert f.shape[2:] == hw, f"got {f.shape[2:]}, expected {hw}"
        assert f.shape[1] == c


def test_pafpn_uniform_channels(device):
    neck = YOLOXPAFPN(in_channels=(64, 128, 256), out_channels=64).to(device)
    p3 = torch.randn(1, 64, 16, 16, device=device)
    p4 = torch.randn(1, 128, 8, 8, device=device)
    p5 = torch.randn(1, 256, 4, 4, device=device)
    out3, out4, out5 = neck((p3, p4, p5))
    assert out3.shape == (1, 64, 16, 16)
    assert out4.shape == (1, 64, 8, 8)
    assert out5.shape == (1, 64, 4, 4)


def test_yolox_head_shapes(device):
    head = YOLOXHeadSNSM(num_classes=2, in_channels=32, feat_channels=32).to(device)
    feats = (
        torch.randn(1, 32, 16, 16, device=device),
        torch.randn(1, 32, 8, 8, device=device),
        torch.randn(1, 32, 4, 4, device=device),
    )
    cls, bbox, obj = head(feats)
    assert len(cls) == len(bbox) == len(obj) == 3
    assert cls[0].shape == (1, 2, 16, 16)
    assert bbox[0].shape == (1, 4, 16, 16)
    assert obj[0].shape == (1, 1, 16, 16)


def test_yolox_head_decode_outputs_format(device):
    head = YOLOXHeadSNSM(num_classes=2, in_channels=32, feat_channels=32).to(device)
    head.eval()
    feats = (
        torch.randn(1, 32, 16, 16, device=device),
        torch.randn(1, 32, 8, 8, device=device),
        torch.randn(1, 32, 4, 4, device=device),
    )
    cls, bbox, obj = head(feats)
    decoded = head.decode_outputs(cls, bbox, obj, img_size=(128, 128))
    n_anchors = 16 * 16 + 8 * 8 + 4 * 4
    assert decoded.shape == (1, n_anchors, 4 + 2)
    # Scores must be in [0, 1] after sigmoid-multiplication
    scores = decoded[..., 4:]
    assert (scores >= 0).all() and (scores <= 1).all()


def test_ccpe_detector_train_eval_modes(device):
    model = CCPE_Detector(
        num_classes=2,
        embed_dims=32,
        depths=(2, 2, 2, 2),
        num_heads=(2, 4, 8, 8),
        window_size=4,
        fpn_channels=32,
        input_size=(128, 128),
        contrast_steps=[1, 2, 4],
    ).to(device)
    x = torch.randn(1, 3, 128, 128, device=device)

    model.train()
    cls, bbox, obj = model(x)
    assert isinstance(cls, list) and len(cls) == 3

    model.eval()
    with torch.no_grad():
        decoded = model(x)
    assert decoded.dim() == 3
    assert decoded.shape[-1] == 4 + 2


def test_ccpe_detector_param_groups(device):
    model = CCPE_Detector(num_classes=2, embed_dims=32,
                         depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 8),
                         window_size=4, fpn_channels=32,
                         input_size=(128, 128), contrast_steps=[1, 2, 4]).to(device)
    pg = model.get_param_groups(lr=1e-4, weight_decay=5e-4)
    assert len(pg) == 2
    assert pg[0]["lr"] == 1e-5  # backbone LR is 0.1× base
    assert pg[1]["lr"] == 1e-4
    # Backbone group should contain Swin params, other group should contain head/neck
    backbone_n = sum(p.numel() for p in pg[0]["params"])
    other_n = sum(p.numel() for p in pg[1]["params"])
    assert backbone_n > 0 and other_n > 0


def test_snsm_sample_negatives(device):
    head = YOLOXHeadSNSM(num_classes=2, in_channels=32, feat_channels=32,
                         pos_sample_rate=4, neg_sample_rate=8).to(device)
    obj_loss = torch.randn(2, 100, device=device).abs()
    has_targets = torch.tensor([True, False], device=device)
    mask = head.sample_negatives(obj_loss, has_targets)
    assert mask.shape == (2, 100)
    assert mask[0].sum().item() == 4   # positive image → pos_sample_rate
    assert mask[1].sum().item() == 8   # negative image → neg_sample_rate
