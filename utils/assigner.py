"""
SimOTA Assignment for YOLOX-style training.
Matches predictions to ground truth using optimal transport.
"""

import torch
import torch.nn.functional as F


class SimOTAAssigner:
    """
    SimOTA assignment strategy from YOLOX.

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
        """
        Assign predictions to ground truth.

        Args:
            pred_scores: (N,) predicted objectness scores
            pred_bboxes: (N, 4) predicted boxes [x1, y1, x2, y2]
            gt_bboxes: (M, 4) ground truth boxes [x1, y1, x2, y2]
            gt_labels: (M,) ground truth class labels
            points: (N, 2) anchor center points
            strides: (N,) stride per anchor
            num_classes: int

        Returns:
            assigned_labels: (N,) assigned class label (-1 for negative)
            assigned_bboxes: (N, 4) assigned GT boxes
            assigned_scores: (N,) assignment quality scores
            fg_mask: (N,) boolean mask for positive anchors
        """
        num_gt = gt_bboxes.shape[0]
        num_pred = pred_bboxes.shape[0]

        if num_gt == 0:
            return (
                torch.full((num_pred,), -1, dtype=torch.long, device=pred_bboxes.device),
                torch.zeros((num_pred, 4), device=pred_bboxes.device),
                torch.zeros(num_pred, device=pred_bboxes.device),
                torch.zeros(num_pred, dtype=torch.bool, device=pred_bboxes.device)
            )

        # Get candidate mask (in GT center region)
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2

        # Check if point is within center_radius * stride of any GT
        distances = torch.cdist(points.float(), torch.stack([gt_cx, gt_cy], dim=1).float())
        is_in_center = distances < (self.center_radius * strides.unsqueeze(1))

        # Also check if point is inside GT box
        lt = points[:, None, :] - gt_bboxes[None, :, :2]  # (N, M, 2)
        rb = gt_bboxes[None, :, 2:] - points[:, None, :]  # (N, M, 2)
        is_in_box = torch.min(torch.cat([lt, rb], dim=-1), dim=-1).values > 0

        # Candidate = in center OR in box
        is_candidate = is_in_center | is_in_box  # (N, M)

        # Compute IoU cost
        ious = self._compute_iou(pred_bboxes, gt_bboxes)  # (N, M)

        # Compute classification cost
        cls_cost = F.binary_cross_entropy(
            pred_scores[:, None].expand(-1, num_gt).clamp(1e-6, 1-1e-6),
            torch.ones(num_pred, num_gt, device=pred_scores.device),
            reduction='none'
        )

        # Total cost
        cost = cls_cost + self.iou_weight * (1 - ious)
        cost[~is_candidate.any(dim=1)] = 1e6

        # Dynamic k selection
        n_candidate = is_candidate.sum(dim=0)  # per GT
        topk = torch.clamp(n_candidate, min=1, max=self.candidate_topk)

        # Assign
        assigned_labels = torch.full((num_pred,), -1, dtype=torch.long, device=pred_bboxes.device)
        assigned_bboxes = torch.zeros((num_pred, 4), device=pred_bboxes.device)
        assigned_scores = torch.zeros(num_pred, device=pred_bboxes.device)

        for gt_idx in range(num_gt):
            candidate_mask = is_candidate[:, gt_idx]
            if not candidate_mask.any():
                continue

            gt_cost = cost[candidate_mask, gt_idx]
            k = min(int(topk[gt_idx].item()), gt_cost.shape[0])
            _, topk_indices = gt_cost.topk(k, largest=False)

            # Map back to full indices
            candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
            selected = candidate_indices[topk_indices]

            assigned_labels[selected] = gt_labels[gt_idx]
            assigned_bboxes[selected] = gt_bboxes[gt_idx]
            assigned_scores[selected] = ious[selected, gt_idx]

        fg_mask = assigned_labels >= 0
        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask

    def _compute_iou(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp(min=1e-6)
