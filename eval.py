"""
Evaluation script for CCPE detector.
Computes COCO mAP and per-class metrics.
"""

import os
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from models import CCPE_Detector
from utils.dataset import FireSmokeDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--conf-thresh', type=float, default=0.25)
    parser.add_argument('--iou-thresh', type=float, default=0.5)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    return parser.parse_args()


def nms(boxes, scores, iou_threshold=0.5):
    """Simple NMS."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def compute_ap(recalls, precisions):
    """Compute AP from recall and precision arrays."""
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([1.], precisions, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate(model, dataloader, num_classes, conf_thresh, iou_thresh, device):
    """Run evaluation and compute mAP."""
    model.eval()
    all_detections = []  # (img_idx, class, conf, x1, y1, x2, y2)
    all_targets = []  # (img_idx, class, x1, y1, x2, y2)

    img_offset = 0
    with torch.no_grad():
        for imgs, targets, indices in dataloader:
            imgs = imgs.to(device)
            with autocast(dtype=torch.bfloat16):
                preds = model(imgs)  # (B, N, 4+num_classes)

            preds = preds.cpu().numpy()
            B = preds.shape[0]

            for b in range(B):
                pred = preds[b]
                # pred: (N, 4+num_classes) — x1,y1,x2,y2,score_per_class
                for cls in range(num_classes):
                    scores = pred[:, 4 + cls]
                    mask = scores > conf_thresh
                    if mask.sum() == 0:
                        continue
                    boxes = pred[mask, :4]
                    cls_scores = scores[mask]
                    keep = nms(boxes, cls_scores, iou_thresh)
                    for k in keep:
                        all_detections.append(
                            (img_offset + b, cls, cls_scores[k],
                             boxes[k, 0], boxes[k, 1], boxes[k, 2], boxes[k, 3])
                        )

                # Ground truth
                tgt = targets[b]
                if tgt.shape[0] > 0:
                    for t in tgt:
                        cls = int(t[0].item())
                        x1, y1, w, h = t[1].item(), t[2].item(), t[3].item(), t[4].item()
                        all_targets.append((img_offset + b, cls, x1, y1, x1 + w, y1 + h))

            img_offset += B

    # Compute per-class AP
    class_names = ['smoke', 'fire']
    results = {}
    for cls in range(num_classes):
        cls_dets = [d for d in all_detections if d[1] == cls]
        cls_gts = [g for g in all_targets if g[1] == cls]

        if len(cls_gts) == 0:
            results[class_names[cls]] = {'AP50': 0.0, 'num_gt': 0}
            continue

        # Sort by confidence
        cls_dets.sort(key=lambda x: -x[2])

        tp = np.zeros(len(cls_dets))
        fp = np.zeros(len(cls_dets))
        gt_matched = set()

        for i, det in enumerate(cls_dets):
            img_idx = det[0]
            det_box = np.array(det[3:7])

            best_iou = 0
            best_gt = -1
            for j, gt in enumerate(cls_gts):
                if gt[0] != img_idx:
                    continue
                gt_box = np.array(gt[2:6])
                iou = compute_iou_single(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j

            if best_iou >= iou_thresh and best_gt not in gt_matched:
                tp[i] = 1
                gt_matched.add(best_gt)
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / len(cls_gts)
        precision = tp_cum / (tp_cum + fp_cum)

        ap = compute_ap(recall, precision)
        results[class_names[cls]] = {
            'AP50': ap,
            'precision': precision[-1] if len(precision) > 0 else 0,
            'recall': recall[-1] if len(recall) > 0 else 0,
            'num_gt': len(cls_gts),
            'num_det': len(cls_dets)
        }

    # mAP
    aps = [r['AP50'] for r in results.values() if r['num_gt'] > 0]
    results['mAP50'] = np.mean(aps) if aps else 0.0
    return results


def compute_iou_single(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(f'cuda:{args.gpu}')

    # Model
    model_cfg = cfg['model']
    model = CCPE_Detector(
        num_classes=model_cfg['num_classes'],
        in_channels=model_cfg.get('in_channels', 3),
        embed_dims=model_cfg.get('embed_dims', 96),
        depths=tuple(model_cfg.get('depths', [2, 2, 6, 2])),
        num_heads=tuple(model_cfg.get('num_heads', [3, 6, 12, 24])),
        window_size=model_cfg.get('window_size', 7),
        fpn_channels=model_cfg.get('fpn_channels', 128),
        input_size=tuple(model_cfg.get('input_size', [1024, 1024])),
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Dataset
    data_cfg = cfg['data']
    if args.split == 'val':
        dataset = FireSmokeDataset(
            img_dir=data_cfg['val_img_dir'],
            label_dir=data_cfg.get('val_label_dir'),
            img_size=model_cfg.get('input_size', [1024, 1024])[0],
            augment=False,
            num_classes=model_cfg['num_classes']
        )
    else:
        # test split — use test_disjoint
        test_img_dir = data_cfg.get('test_img_dir', data_cfg['val_img_dir'].replace('test_random', 'test_disjoint'))
        test_label_dir = data_cfg.get('test_label_dir', data_cfg.get('val_label_dir', '').replace('test_random', 'test_disjoint'))
        dataset = FireSmokeDataset(
            img_dir=test_img_dir,
            label_dir=test_label_dir,
            img_size=model_cfg.get('input_size', [1024, 1024])[0],
            augment=False,
            num_classes=model_cfg['num_classes']
        )

    dataloader = DataLoader(
        dataset, batch_size=8, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )

    print(f"Evaluating on {len(dataset)} images ({args.split} split)")
    print(f"Confidence threshold: {args.conf_thresh}")

    results = evaluate(model, dataloader, model_cfg['num_classes'],
                       args.conf_thresh, args.iou_thresh, device)

    print("\n" + "=" * 50)
    print(f"{'Class':<10} {'AP50':<8} {'P':<8} {'R':<8} {'GT':<6} {'Det':<6}")
    print("-" * 50)
    for cls_name in ['smoke', 'fire']:
        if cls_name in results:
            r = results[cls_name]
            print(f"{cls_name:<10} {r['AP50']:.4f}  {r.get('precision', 0):.4f}  "
                  f"{r.get('recall', 0):.4f}  {r.get('num_gt', 0):<6} {r.get('num_det', 0):<6}")
    print("-" * 50)
    print(f"{'mAP50':<10} {results['mAP50']:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
