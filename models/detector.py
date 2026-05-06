"""
CCPE Detector — Full model combining backbone, neck, and head.
"""

import torch
import torch.nn as nn
from .swin_ccpe import SwinTransformerCCPE
from .neck import YOLOXPAFPN
from .head import YOLOXHeadSNSM


class CCPE_Detector(nn.Module):
    """
    Full CCPE detection model:
    - Backbone: Swin Transformer + Cross Contrast Patch Embedding
    - Neck: YOLOX PAFPN
    - Head: YOLOX head with Separable Negative Sampling

    Args:
        num_classes: number of detection classes
        in_channels: input image channels (3 for RGB, 6 for multi-frame)
        embed_dims: Swin embedding dimension (96 for Tiny, 128 for Base)
        depths: Swin stage depths
        num_heads: attention heads per stage
        window_size: Swin window size
        fpn_channels: PAFPN output channels
        input_size: expected input image size (H, W)
    """

    def __init__(self, num_classes=2, in_channels=3, embed_dims=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, fpn_channels=128, input_size=(1024, 1024),
                 contrast_steps=None, use_checkpoint=False,
                 pretrained_swin=None):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # Backbone
        self.backbone = SwinTransformerCCPE(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            out_indices=(1, 2, 3),
            use_checkpoint=use_checkpoint,
            contrast_steps=contrast_steps,
            pretrained=pretrained_swin
        )

        # Neck — channels from Swin stages 1,2,3
        backbone_channels = [embed_dims * 2, embed_dims * 4, embed_dims * 8]
        self.neck = YOLOXPAFPN(
            in_channels=backbone_channels,
            out_channels=fpn_channels,
            num_csp_blocks=1
        )

        # Head
        self.head = YOLOXHeadSNSM(
            num_classes=num_classes,
            in_channels=fpn_channels,
            feat_channels=fpn_channels,
            stacked_convs=2,
            strides=(8, 16, 32)
        )

        if pretrained_swin:
            self.backbone.load_pretrained(pretrained_swin)

    def forward(self, x, return_raw=False):
        """
        Args:
            x: (B, C, H, W) input images
            return_raw: if True, always return raw (cls_scores, bbox_preds,
                obj_scores) regardless of train/eval mode. Needed by the
                validate() loop, which puts the model in eval() (to disable
                Dropout/DropPath) but still needs raw logits to compute val
                loss with the same loss_fn used during training.

        Returns:
            In training mode (or return_raw=True):
                cls_scores, bbox_preds, obj_scores  (raw logits per FPN level)
            In eval mode (return_raw=False):
                decoded predictions (B, N, 4 + 1 + num_classes)
        """
        features = self.backbone(x)
        fpn_features = self.neck(features)
        cls_scores, bbox_preds, obj_scores = self.head(fpn_features)

        if not self.training and not return_raw:
            return self.head.decode_outputs(
                cls_scores, bbox_preds, obj_scores, self.input_size
            )

        return cls_scores, bbox_preds, obj_scores

    @property
    def device(self):
        return next(self.parameters()).device

    def get_param_groups(self, lr, weight_decay):
        """Get parameter groups with different LR for backbone vs head."""
        backbone_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        return [
            {'params': backbone_params, 'lr': lr * 0.1, 'weight_decay': weight_decay},
            {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
        ]
