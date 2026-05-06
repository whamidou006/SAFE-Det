"""
YOLOX Detection Head with Separable Negative Sampling Mechanism (SNSM).

Key innovation: Different negative sampling strategies for positive images
(containing objects) vs negative images (background only).
- Positive images: Random sampling (rate=10)
- Negative images: OHEM sampling (rate=190, emphasizes hard negatives)

This addresses supervision signal confusion in smoke detection where
background patches in positive images and negative images have different
significance for learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YOLOXHeadSNSM(nn.Module):
    """
    YOLOX detection head with Separable Negative Sampling.

    Produces: cls_scores, bbox_preds, objectness at 3 FPN scales.
    Training uses SimOTA assignment + SNSM.
    """

    def __init__(self, num_classes=2, in_channels=128, feat_channels=128,
                 stacked_convs=2, strides=(8, 16, 32),
                 pos_sample_rate=10, neg_sample_rate=190,
                 neg_mu=-300000, neg_mse=500000):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.pos_sample_rate = pos_sample_rate
        self.neg_sample_rate = neg_sample_rate
        self.neg_mu = neg_mu
        self.neg_mse = neg_mse

        # Build heads for each scale
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for _ in strides:
            cls_conv = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else feat_channels,
                              feat_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(feat_channels),
                    nn.SiLU(inplace=True)
                ) for i in range(stacked_convs)
            ])
            reg_conv = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else feat_channels,
                              feat_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(feat_channels),
                    nn.SiLU(inplace=True)
                ) for i in range(stacked_convs)
            ])
            self.cls_convs.append(cls_conv)
            self.reg_convs.append(reg_conv)
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, 1))
            self.obj_preds.append(nn.Conv2d(feat_channels, 1, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Initialize cls bias for better convergence
        for pred in self.cls_preds:
            nn.init.constant_(pred.bias, -math.log((1 - 0.01) / 0.01))
        for pred in self.obj_preds:
            nn.init.constant_(pred.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, features):
        """
        Args:
            features: tuple of feature maps from neck (P3, P4, P5)

        Returns:
            cls_scores: list of (B, num_classes, H, W) per scale
            bbox_preds: list of (B, 4, H, W) per scale
            obj_scores: list of (B, 1, H, W) per scale
        """
        cls_scores = []
        bbox_preds = []
        obj_scores = []

        for i, feat in enumerate(features):
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)

            cls_scores.append(self.cls_preds[i](cls_feat))
            bbox_preds.append(self.reg_preds[i](reg_feat))
            obj_scores.append(self.obj_preds[i](reg_feat))

        return cls_scores, bbox_preds, obj_scores

    def decode_outputs(self, cls_scores, bbox_preds, obj_scores, img_size):
        """Decode raw outputs into final predictions."""
        all_preds = []
        for i, stride in enumerate(self.strides):
            B, _, H, W = cls_scores[i].shape
            # Generate grid
            yv, xv = torch.meshgrid(
                torch.arange(H, device=cls_scores[i].device, dtype=torch.float32),
                torch.arange(W, device=cls_scores[i].device, dtype=torch.float32),
                indexing='ij'
            )
            grid = torch.stack([xv, yv], dim=-1).view(1, H * W, 2)

            # Decode boxes (center + wh)
            bbox = bbox_preds[i].permute(0, 2, 3, 1).reshape(B, H * W, 4)
            xy = (bbox[..., :2] + grid) * stride
            wh = bbox[..., 2:].exp() * stride

            # Decode scores
            cls = cls_scores[i].permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)
            obj = obj_scores[i].permute(0, 2, 3, 1).reshape(B, H * W, 1)
            scores = cls.sigmoid() * obj.sigmoid()

            # Convert to x1y1x2y2
            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            preds = torch.cat([x1y1, x2y2, scores], dim=-1)
            all_preds.append(preds)

        return torch.cat(all_preds, dim=1)

    def sample_negatives(self, obj_loss_per_anchor, has_targets):
        """
        Separable Negative Sampling Mechanism (SNSM).

        For positive images: random sample negatives (rate=pos_sample_rate)
        For negative images: OHEM sample negatives (rate=neg_sample_rate)

        Args:
            obj_loss_per_anchor: (B, N) objectness loss per anchor
            has_targets: (B,) bool indicating which images have GT boxes

        Returns:
            neg_mask: (B, N) bool mask of selected negative anchors
        """
        B, N = obj_loss_per_anchor.shape
        neg_mask = torch.zeros_like(obj_loss_per_anchor, dtype=torch.bool)

        for b in range(B):
            if has_targets[b]:
                # Positive image: random sampling
                num_neg = min(self.pos_sample_rate, N)
                indices = torch.randperm(N, device=obj_loss_per_anchor.device)[:num_neg]
                neg_mask[b, indices] = True
            else:
                # Negative image: OHEM — select hardest negatives
                num_neg = min(self.neg_sample_rate, N)
                # Gaussian weighting for OHEM
                weights = torch.exp(
                    -(obj_loss_per_anchor[b] - self.neg_mu) ** 2 / (2 * self.neg_mse)
                )
                _, topk_indices = weights.topk(num_neg)
                neg_mask[b, topk_indices] = True

        return neg_mask
