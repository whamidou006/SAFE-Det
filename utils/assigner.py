"""
Label assignment strategies for SAFE-Det.

Three options are exposed via :func:`build_assigner` and selected from the
YAML config (``loss.assigner``):

- ``'simota'``  — YOLOX SimOTA (existing default; behaviour preserved)
- ``'tal'``    — Task-Aligned Learning (TOOD, ICCV 2021)
- ``'dsla'``   — Dynamic Soft Label Assignment (RTMDet, 2022)

All three implement the same interface so they can be swapped behind a
config flag without changing the loss code:

    assigned_labels, assigned_bboxes, assigned_scores, fg_mask = (
        assigner.assign(pred_scores, pred_bboxes, gt_bboxes, gt_labels,
                        points, strides, num_classes)
    )

``assigned_scores`` is the soft target used by the cls and bbox losses.
For SimOTA it is the IoU of the matched pair (current behaviour). For
TAL it is the alignment metric ``(cls_score)^alpha * iou^beta`` and for
DSLA it is the IoU itself (RTMDet uses IoU as the soft cls label).

References:
- SimOTA — Ge et al., "YOLOX", arXiv:2107.08430 (2021).
- TAL    — Feng et al., "TOOD", ICCV 2021.
- DSLA   — Lyu et al., "RTMDet", arXiv:2212.07784 (2022).
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=eps)


def _empty_assignment(num_pred: int, device):
    return (
        torch.full((num_pred,), -1, dtype=torch.long, device=device),
        torch.zeros((num_pred, 4), device=device),
        torch.zeros(num_pred, device=device),
        torch.zeros(num_pred, dtype=torch.bool, device=device),
    )


def _candidate_mask(points: torch.Tensor, strides: torch.Tensor,
                    gt_bboxes: torch.Tensor, center_radius: float) -> torch.Tensor:
    """Return (N, M) bool mask of (point, GT) pairs that are either inside
    the GT box or within ``center_radius * stride`` of its centre."""
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
    distances = torch.cdist(
        points.float(), torch.stack([gt_cx, gt_cy], dim=1).float()
    )
    is_in_center = distances < (center_radius * strides.unsqueeze(1))
    lt = points[:, None, :] - gt_bboxes[None, :, :2]
    rb = gt_bboxes[None, :, 2:] - points[:, None, :]
    is_in_box = torch.min(torch.cat([lt, rb], dim=-1), dim=-1).values > 0
    return is_in_center | is_in_box


# --------------------------------------------------------------------------- #
# SimOTA (existing — kept verbatim for backward compatibility)                #
# --------------------------------------------------------------------------- #


class SimOTAAssigner:
    """SimOTA assignment strategy from YOLOX.

    For each GT box, selects top-k predictions as candidates,
    then uses Sinkhorn-like cost minimization to find optimal assignment.
    """

    def __init__(self, center_radius=2.5, candidate_topk=10, iou_weight=3.0):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, gt_bboxes, gt_labels,
               points, strides, num_classes):
        num_gt = gt_bboxes.shape[0]
        num_pred = pred_bboxes.shape[0]
        if num_gt == 0:
            return _empty_assignment(num_pred, pred_bboxes.device)

        is_candidate = _candidate_mask(points, strides, gt_bboxes,
                                       self.center_radius)
        ious = _pairwise_iou(pred_bboxes, gt_bboxes)

        cls_cost = F.binary_cross_entropy(
            pred_scores[:, None].expand(-1, num_gt).clamp(1e-6, 1 - 1e-6),
            torch.ones(num_pred, num_gt, device=pred_scores.device),
            reduction='none',
        )
        cost = cls_cost + self.iou_weight * (1 - ious)
        cost[~is_candidate.any(dim=1)] = 1e6

        n_candidate = is_candidate.sum(dim=0)
        topk = torch.clamp(n_candidate, min=1, max=self.candidate_topk)

        assigned_labels = torch.full((num_pred,), -1, dtype=torch.long,
                                     device=pred_bboxes.device)
        assigned_bboxes = torch.zeros((num_pred, 4), device=pred_bboxes.device)
        assigned_scores = torch.zeros(num_pred, device=pred_bboxes.device)

        for gt_idx in range(num_gt):
            cmask = is_candidate[:, gt_idx]
            if not cmask.any():
                continue
            gt_cost = cost[cmask, gt_idx]
            k = min(int(topk[gt_idx].item()), gt_cost.shape[0])
            _, ti = gt_cost.topk(k, largest=False)
            cand_idx = cmask.nonzero(as_tuple=True)[0]
            sel = cand_idx[ti]
            assigned_labels[sel] = gt_labels[gt_idx]
            assigned_bboxes[sel] = gt_bboxes[gt_idx]
            assigned_scores[sel] = ious[sel, gt_idx]

        fg_mask = assigned_labels >= 0
        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask


# --------------------------------------------------------------------------- #
# TAL — Task-Aligned Learning (TOOD, ICCV 2021)                               #
# --------------------------------------------------------------------------- #


class TALAssigner:
    """Task-Aligned Learning assignment.

    For every (anchor, GT) pair compute an alignment metric
    ``t = sigmoid(cls)^alpha * iou^beta`` and pick the top-``topk`` anchors
    per GT (subject to a centre-prior gate). The alignment metric itself
    is also returned as ``assigned_scores`` and is intended to be used as
    a soft cls target by the loss.

    Defaults follow the TOOD paper (alpha=1.0, beta=6.0, topk=13).
    """

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0,
                 center_radius: float = 2.5):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_radius = center_radius

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, gt_bboxes, gt_labels,
               points, strides, num_classes):
        num_gt = gt_bboxes.shape[0]
        num_pred = pred_bboxes.shape[0]
        if num_gt == 0:
            return _empty_assignment(num_pred, pred_bboxes.device)

        is_candidate = _candidate_mask(points, strides, gt_bboxes,
                                       self.center_radius)
        ious = _pairwise_iou(pred_bboxes, gt_bboxes)
        # pred_scores is the per-anchor objectness probability already in [0,1].
        align = (pred_scores[:, None].clamp(1e-6, 1 - 1e-6) ** self.alpha) \
                * (ious.clamp(min=0) ** self.beta)
        align = align * is_candidate.float()

        assigned_labels = torch.full((num_pred,), -1, dtype=torch.long,
                                     device=pred_bboxes.device)
        assigned_bboxes = torch.zeros((num_pred, 4), device=pred_bboxes.device)
        assigned_scores = torch.zeros(num_pred, device=pred_bboxes.device)

        for gt_idx in range(num_gt):
            scores_g = align[:, gt_idx]
            if (scores_g > 0).sum() == 0:
                continue
            k = min(self.topk, int((scores_g > 0).sum().item()))
            _, top_idx = scores_g.topk(k, largest=True)
            assigned_labels[top_idx] = gt_labels[gt_idx]
            assigned_bboxes[top_idx] = gt_bboxes[gt_idx]
            # store the alignment metric as the soft target
            assigned_scores[top_idx] = scores_g[top_idx]

        fg_mask = assigned_labels >= 0
        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask


# --------------------------------------------------------------------------- #
# DSLA — Dynamic Soft Label Assignment (RTMDet, 2022)                         #
# --------------------------------------------------------------------------- #


class DSLAAssigner:
    """RTMDet-style dynamic soft label assignment.

    Cost = cls_cost + iou_weight * (1 - IoU) + center_dist_cost.
    Dynamic ``k`` per GT is the integer sum of the top-10 IoU values of its
    candidate anchors (clamped to >= 1). The IoU itself is returned as the
    soft cls target — the RTMDet paper shows this beats hard one-hot.
    """

    def __init__(self, topq: int = 13, iou_weight: float = 3.0,
                 center_dist_weight: float = 1.0, center_radius: float = 2.5):
        self.topq = topq
        self.iou_weight = iou_weight
        self.center_dist_weight = center_dist_weight
        self.center_radius = center_radius

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, gt_bboxes, gt_labels,
               points, strides, num_classes):
        num_gt = gt_bboxes.shape[0]
        num_pred = pred_bboxes.shape[0]
        if num_gt == 0:
            return _empty_assignment(num_pred, pred_bboxes.device)

        is_candidate = _candidate_mask(points, strides, gt_bboxes,
                                       self.center_radius)
        ious = _pairwise_iou(pred_bboxes, gt_bboxes)

        cls_cost = F.binary_cross_entropy(
            pred_scores[:, None].expand(-1, num_gt).clamp(1e-6, 1 - 1e-6),
            torch.ones(num_pred, num_gt, device=pred_scores.device),
            reduction='none',
        )

        # Centre distance cost (normalised by GT diagonal length).
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
        gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
        diag = torch.sqrt(gt_w ** 2 + gt_h ** 2)
        gt_centers = torch.stack([gt_cx, gt_cy], dim=1)
        center_dist = torch.cdist(points.float(), gt_centers.float()) / diag[None]

        cost = cls_cost + self.iou_weight * (1 - ious) \
               + self.center_dist_weight * center_dist
        cost[~is_candidate] = 1e6

        # Dynamic k per GT.
        topq = min(self.topq, num_pred)
        topk_iou, _ = ious.topk(topq, dim=0)
        dyn_k = topk_iou.sum(dim=0).clamp(min=1).long()

        assigned_labels = torch.full((num_pred,), -1, dtype=torch.long,
                                     device=pred_bboxes.device)
        assigned_bboxes = torch.zeros((num_pred, 4), device=pred_bboxes.device)
        assigned_scores = torch.zeros(num_pred, device=pred_bboxes.device)

        for gt_idx in range(num_gt):
            cmask = is_candidate[:, gt_idx]
            if not cmask.any():
                continue
            gt_cost = cost[cmask, gt_idx]
            k = min(int(dyn_k[gt_idx].item()), gt_cost.shape[0])
            _, ti = gt_cost.topk(k, largest=False)
            cand_idx = cmask.nonzero(as_tuple=True)[0]
            sel = cand_idx[ti]
            assigned_labels[sel] = gt_labels[gt_idx]
            assigned_bboxes[sel] = gt_bboxes[gt_idx]
            # soft cls target = IoU (RTMDet recipe)
            assigned_scores[sel] = ious[sel, gt_idx]

        fg_mask = assigned_labels >= 0
        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def build_assigner(name: str = 'simota', **kwargs):
    """Construct an assigner by name. Default ``'simota'`` preserves
    historical behaviour exactly."""
    name = name.lower()
    if name == 'simota':
        return SimOTAAssigner(**kwargs)
    if name == 'tal':
        return TALAssigner(**kwargs)
    if name == 'dsla':
        return DSLAAssigner(**kwargs)
    raise ValueError(
        f"Unknown assigner '{name}'. Choose one of: 'simota', 'tal', 'dsla'."
    )
