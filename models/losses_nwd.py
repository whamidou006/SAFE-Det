"""
Normalized Wasserstein Distance (NWD) for tiny object detection.

Reference:
    Wang, Xu, Yang, Yu (2021/2022). *A Normalized Gaussian Wasserstein
    Distance for Tiny Object Detection.* arXiv:2110.13389. Extended in
    ISPRS Journal of Photogrammetry and Remote Sensing, 2022.
    Code: https://github.com/jwwangchn/NWD

Why this exists in SAFE-Det:
    Smoke onset wisps are sub-pixel/few-pixel. IoU is mathematically
    degenerate for boxes <16 px (a 1-pixel jitter swings IoU 0 -> 0.6),
    so SimOTA's IoU prefilter starves them of positive samples and the
    bbox regressor receives no gradient.

    NWD models each box as a 2-D Gaussian and computes the 2-Wasserstein
    distance between predicted and GT distributions. This stays smooth
    and informative for tiny boxes.

Usage:
    Opt in via the YAML config:

        loss:
          bbox_loss_type: nwd       # one of: ciou (default) | nwd | mixed
          nwd_constant: 12.8        # dataset-dependent scale; 12.8 is the
                                    #   AI-TOD value the NWD paper uses
          nwd_mix_weight: 0.5       # only used when bbox_loss_type == mixed

The default behaviour is unchanged (CIoU/IoU), so this is a pure
additive option.
"""

import torch


def nwd_iou(pred_xyxy: torch.Tensor,
            target_xyxy: torch.Tensor,
            constant: float = 12.8,
            eps: float = 1e-7) -> torch.Tensor:
    """Normalized Wasserstein Distance between two sets of axis-aligned boxes.

    Each box is modelled as a 2-D Gaussian with mean = box centre and
    diagonal covariance = (w/2)^2, (h/2)^2. Returns a similarity score
    in (0, 1] (1 = identical), so it can be used as a drop-in for IoU.

    Args:
        pred_xyxy:   (N, 4) tensor in [x1, y1, x2, y2] format.
        target_xyxy: (N, 4) tensor in [x1, y1, x2, y2] format.
        constant:    Normalisation scale. Per the paper, set to the
                     average GT box size of the dataset (12.8 for AI-TOD).
                     For wildfire smoke at 1024 input we use 12.8 as the
                     default; tune downward for tinier-than-AI-TOD targets.
        eps:         Numerical floor to avoid division by zero on
                     degenerate boxes.

    Returns:
        (N,) tensor of NWD similarity values in (0, 1].
    """
    pred_cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) * 0.5
    pred_cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) * 0.5
    pred_w = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=eps)
    pred_h = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=eps)

    tgt_cx = (target_xyxy[:, 0] + target_xyxy[:, 2]) * 0.5
    tgt_cy = (target_xyxy[:, 1] + target_xyxy[:, 3]) * 0.5
    tgt_w = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=eps)
    tgt_h = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=eps)

    # Squared 2-Wasserstein distance between two 2-D Gaussians with
    # diagonal covariance: ||mu_p - mu_t||^2 + ||sigma_p - sigma_t||_F^2
    centre_term = (pred_cx - tgt_cx).pow(2) + (pred_cy - tgt_cy).pow(2)
    scale_term = ((pred_w - tgt_w) * 0.5).pow(2) + ((pred_h - tgt_h) * 0.5).pow(2)
    w2 = centre_term + scale_term

    # Normalised exponential similarity in (0, 1].
    return torch.exp(-torch.sqrt(w2.clamp(min=eps)) / constant)


def bbox_loss(pred_xyxy: torch.Tensor,
              target_xyxy: torch.Tensor,
              loss_type: str = 'ciou',
              nwd_constant: float = 12.8,
              nwd_mix_weight: float = 0.5,
              eps: float = 1e-7) -> torch.Tensor:
    """Unified per-box regression loss dispatcher.

    Args:
        pred_xyxy:      (N, 4) predicted boxes in [x1, y1, x2, y2].
        target_xyxy:    (N, 4) target boxes in [x1, y1, x2, y2].
        loss_type:      'ciou' (= 1 - IoU, current default), 'nwd'
                        (= 1 - NWD), or 'mixed'
                        (= w*(1 - IoU) + (1 - w)*(1 - NWD)).
        nwd_constant:   See `nwd_iou`.
        nwd_mix_weight: w in the 'mixed' formula. Default 0.5.
        eps:            Numerical floor.

    Returns:
        (N,) tensor of per-box losses (caller sums and normalises by num_fg).
    """
    if loss_type == 'ciou':
        return _iou_loss(pred_xyxy, target_xyxy, eps=eps)
    if loss_type == 'nwd':
        return 1.0 - nwd_iou(pred_xyxy, target_xyxy, constant=nwd_constant, eps=eps)
    if loss_type == 'mixed':
        iou_part = _iou_loss(pred_xyxy, target_xyxy, eps=eps)
        nwd_part = 1.0 - nwd_iou(pred_xyxy, target_xyxy, constant=nwd_constant, eps=eps)
        return nwd_mix_weight * iou_part + (1.0 - nwd_mix_weight) * nwd_part
    raise ValueError(
        f"Unknown bbox_loss_type='{loss_type}'. "
        "Expected one of: 'ciou', 'nwd', 'mixed'."
    )


def _iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Plain 1 - IoU. Identical to YOLOXLoss._iou_loss; centralised here so
    the dispatcher in `bbox_loss` is self-contained."""
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_pred + area_target - inter
    iou = inter / union.clamp(min=eps)
    return 1.0 - iou
