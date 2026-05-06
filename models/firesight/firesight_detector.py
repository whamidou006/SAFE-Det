"""
FireSight Detector — Novel architecture for fire/smoke detection.

Combines:
- DINOv2 backbone (or Swin with CCPE for comparison)
- Smoke-Aware Feature Enhancement (SAFE): DCM + FAM + Transparency
- Temporal Motion Fusion (optional)
- Hybrid detection head (DETR or YOLOX+SNSM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deformable_contrast import DeformableContrastModule
from .frequency_attention import FrequencyAttentionModule
from .transparency import TransparencyModule
from .temporal_fusion import TemporalMotionFusion


class SAFEModule(nn.Module):
    """
    Smoke-Aware Feature Enhancement.
    Applied to multi-scale features from backbone.

    Combines:
    - Deformable Contrast (DCM): learned spatial gradients
    - Frequency Attention (FAM): frequency-domain reweighting
    - Transparency Modeling (TM): background subtraction in feature space
    """

    def __init__(self, channels: int, use_dcm=True, use_fam=True,
                 use_tm=True, num_points=8):
        super().__init__()
        self.use_dcm = use_dcm
        self.use_fam = use_fam
        self.use_tm = use_tm

        if use_dcm:
            self.dcm = DeformableContrastModule(channels, num_points=num_points)
        if use_fam:
            self.fam = FrequencyAttentionModule(channels, num_freq_bands=4)
        if use_tm:
            self.tm = TransparencyModule(channels, num_scales=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dcm:
            x = self.dcm(x)
        if self.use_fam:
            x = self.fam(x)
        if self.use_tm:
            x = self.tm(x)
        return x


class DINOv2Backbone(nn.Module):
    """
    DINOv2 ViT as backbone with FPN adapter for multi-scale features.

    Uses intermediate layers of DINOv2 to produce multi-scale features:
    - P3 (stride 8): from early layers
    - P4 (stride 16): from middle layers (native ViT stride)
    - P5 (stride 32): from late layers with downsampling
    """

    def __init__(self, model_name='dinov2_vits14', out_channels=256,
                 freeze_backbone=False):
        super().__init__()
        self.out_channels = out_channels

        # Load DINOv2 (will download if not cached)
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name,
                                           pretrained=True)
        except Exception:
            # Fallback: create ViT-S manually
            from torchvision.models import vit_b_16
            self.backbone = vit_b_16(pretrained=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get embed dim from model
        embed_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 384

        # FPN adapters: project DINOv2 features to uniform channels
        self.adapter_p3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, out_channels, 2, stride=2),  # 2x upsample
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.adapter_p4 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.adapter_p5 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image (H,W should be divisible by 14)

        Returns:
            (P3, P4, P5) multi-scale features
        """
        B, C, H, W = x.shape

        # Get intermediate features from DINOv2
        # DINOv2 outputs patch tokens at stride 14
        if hasattr(self.backbone, 'get_intermediate_layers'):
            # Official DINOv2 API
            features = self.backbone.get_intermediate_layers(x, n=[4, 8, 12])
            # Each is (B, N_patches, embed_dim)
            patch_h, patch_w = H // 14, W // 14
            feat_maps = [f.reshape(B, patch_h, patch_w, -1).permute(0, 3, 1, 2)
                         for f in features]
        else:
            # Fallback: single forward, use output
            feat = self.backbone.forward_features(x) if hasattr(self.backbone, 'forward_features') else self.backbone(x)
            if isinstance(feat, dict):
                feat = feat['last_hidden_state']
            # Remove CLS token if present
            if feat.shape[1] > (H // 14) * (W // 14):
                feat = feat[:, 1:]
            patch_h, patch_w = H // 14, W // 14
            feat = feat.reshape(B, patch_h, patch_w, -1).permute(0, 3, 1, 2)
            feat_maps = [feat, feat, feat]

        # Apply FPN adapters
        p3 = self.adapter_p3(feat_maps[0])  # stride 7 (14/2)
        p4 = self.adapter_p4(feat_maps[1])  # stride 14
        p5 = self.adapter_p5(feat_maps[2])  # stride 28

        return (p3, p4, p5)


class FireSightDetector(nn.Module):
    """
    FireSight: Novel Fire/Smoke Detection Architecture.

    Architecture:
    1. Backbone: DINOv2 (or Swin+CCPE) → multi-scale features
    2. SAFE: Smoke-Aware Feature Enhancement at each scale
    3. Temporal: Motion-guided cross-frame fusion (optional)
    4. Neck: PAFPN for feature pyramid fusion
    5. Head: YOLOX or DETR-style

    Args:
        num_classes: detection classes (2: smoke, fire)
        backbone_type: 'dinov2' or 'swin_ccpe'
        use_dcm: enable Deformable Contrast Module
        use_fam: enable Frequency Attention Module
        use_tm: enable Transparency Module
        use_temporal: enable temporal fusion
        head_type: 'yolox' or 'detr'
    """

    def __init__(self, num_classes=2, backbone_type='dinov2',
                 backbone_channels=256, use_dcm=True, use_fam=True,
                 use_tm=True, use_temporal=False, head_type='yolox',
                 freeze_backbone=False, input_size=(1024, 1024)):
        super().__init__()
        self.input_size = input_size
        self.use_temporal = use_temporal
        self.backbone_type = backbone_type

        # Backbone
        if backbone_type == 'dinov2':
            self.backbone = DINOv2Backbone(
                out_channels=backbone_channels,
                freeze_backbone=freeze_backbone
            )
            fpn_in_channels = [backbone_channels] * 3
        else:
            # Use Swin+CCPE backbone (from parent module)
            from ..swin_ccpe import SwinTransformerCCPE
            self.backbone = SwinTransformerCCPE(
                embed_dims=96, depths=(2, 2, 6, 2),
                num_heads=(3, 6, 12, 24), out_indices=(1, 2, 3)
            )
            fpn_in_channels = [192, 384, 768]
            backbone_channels = 192  # smallest scale

        # SAFE modules (applied per scale)
        self.safe_modules = nn.ModuleList([
            SAFEModule(ch, use_dcm=use_dcm, use_fam=use_fam, use_tm=use_tm)
            for ch in fpn_in_channels
        ])

        # Temporal fusion (applied to smallest scale only for efficiency)
        if use_temporal:
            self.temporal = TemporalMotionFusion(
                fpn_in_channels[0], num_heads=4, num_layers=2
            )

        # Neck (PAFPN)
        from ..neck import YOLOXPAFPN
        fpn_out = backbone_channels // 2 if backbone_type == 'dinov2' else 128
        self.neck = YOLOXPAFPN(
            in_channels=fpn_in_channels,
            out_channels=fpn_out,
            num_csp_blocks=1
        )

        # Head
        if head_type == 'yolox':
            from ..head import YOLOXHeadSNSM
            self.head = YOLOXHeadSNSM(
                num_classes=num_classes,
                in_channels=fpn_out,
                feat_channels=fpn_out,
                stacked_convs=2
            )
        else:
            raise NotImplementedError("DETR head not yet implemented")

        self.head_type = head_type

    def forward(self, x, prev_x=None):
        """
        Args:
            x: (B, 3, H, W) current frame
            prev_x: (B, 3, H, W) previous frame (optional, for temporal)

        Returns:
            Training: cls_scores, bbox_preds, obj_scores
            Eval: decoded predictions
        """
        # Backbone
        features = self.backbone(x)

        # SAFE enhancement
        enhanced = []
        for i, (feat, safe) in enumerate(zip(features, self.safe_modules)):
            enhanced.append(safe(feat))

        # Temporal fusion (on first/highest-res scale)
        if self.use_temporal and prev_x is not None:
            prev_features = self.backbone(prev_x)
            enhanced[0] = self.temporal(enhanced[0], prev_features[0])

        # Neck
        fpn_out = self.neck(tuple(enhanced))

        # Head
        cls_scores, bbox_preds, obj_scores = self.head(fpn_out)

        if not self.training:
            return self.head.decode_outputs(cls_scores, bbox_preds, obj_scores, self.input_size)

        return cls_scores, bbox_preds, obj_scores
