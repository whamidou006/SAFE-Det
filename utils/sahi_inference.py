"""
SAHI (Slicing Aided Hyper Inference) integration for SAFE-Det.

Enables sliced inference for detecting small smoke/fire in large images.
Instead of resizing a 1920×1080 image to 1024×1024 (losing detail),
SAHI tiles the image into overlapping slices, runs detection on each,
and merges results with NMS.

This is especially valuable for:
- Early smoke detection (small wisps at distance)
- High-resolution camera feeds (1920×1080, 4K)
- Objects near the horizon line (appear very small)

Usage:
    python eval.py --config configs/ccpe_single_1024.yaml \
                   --checkpoint runs/best.pth \
                   --sahi --slice-size 640 --overlap 0.2

Requirements:
    pip install sahi
"""

import numpy as np
from pathlib import Path

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction, get_prediction
    from sahi.models.base import DetectionModel
    from sahi.prediction import ObjectPrediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


def check_sahi_available():
    if not SAHI_AVAILABLE:
        raise ImportError(
            "SAHI not installed. Install with: pip install sahi\n"
            "SAHI enables sliced inference for better small object detection."
        )


class SAFEDetSahiModel(DetectionModel if SAHI_AVAILABLE else object):
    """
    SAHI-compatible wrapper for SAFE-Det models.

    Wraps our CCPE/FireSight detector to work with SAHI's sliced inference.
    SAHI handles the image slicing, and this class handles per-slice detection.
    """

    def __init__(self, model, device, num_classes=2,
                 class_names=None, conf_thresh=0.25, **kwargs):
        """
        Args:
            model: SAFE-Det model (CCPE_Detector or FireSightDetector)
            device: torch device
            num_classes: number of classes
            class_names: list of class names
            conf_thresh: confidence threshold
        """
        self._model = model
        self._device = device
        self._num_classes = num_classes
        self._class_names = class_names or ['smoke', 'fire']
        self._conf_thresh = conf_thresh

        # Required by SAHI base class
        if SAHI_AVAILABLE:
            self.model = model
            self.num_categories = num_classes
            self.category_names = self._class_names
            self.category_mapping = {
                str(i): name for i, name in enumerate(self._class_names)
            }

    @property
    def num_categories(self):
        return self._num_classes

    @num_categories.setter
    def num_categories(self, value):
        self._num_classes = value

    @property
    def category_names(self):
        return self._class_names

    @category_names.setter
    def category_names(self, value):
        self._class_names = value

    @property
    def category_mapping(self):
        return self._category_mapping

    @category_mapping.setter
    def category_mapping(self, value):
        self._category_mapping = value

    def load_model(self):
        """Required by SAHI — model is already loaded."""
        pass

    def perform_inference(self, image: np.ndarray):
        """
        Run inference on a single image (numpy array, BGR/RGB, HWC).

        Args:
            image: numpy array (H, W, 3)

        Sets:
            self._original_predictions: raw model output
        """
        import torch
        import cv2

        h, w = image.shape[:2]
        img_size = self._model.input_size[0] if hasattr(self._model, 'input_size') else 1024

        # Resize with letterbox
        scale = min(img_size / h, img_size / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (nw, nh))
        padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
        dy, dx = (img_size - nh) // 2, (img_size - nw) // 2
        padded[dy:dy + nh, dx:dx + nw] = resized

        # To tensor
        img_tensor = torch.from_numpy(
            padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0).to(self._device)

        # Forward
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            preds = self._model(img_tensor)  # (1, N, 4+num_classes)

        preds = preds[0].cpu().numpy()  # (N, 4+num_classes)

        # Scale predictions back to original image coordinates
        preds[:, 0] = (preds[:, 0] - dx) / scale  # x1
        preds[:, 1] = (preds[:, 1] - dy) / scale  # y1
        preds[:, 2] = (preds[:, 2] - dx) / scale  # x2
        preds[:, 3] = (preds[:, 3] - dy) / scale  # y2

        # Clip to image bounds
        preds[:, 0] = np.clip(preds[:, 0], 0, w)
        preds[:, 1] = np.clip(preds[:, 1], 0, h)
        preds[:, 2] = np.clip(preds[:, 2], 0, w)
        preds[:, 3] = np.clip(preds[:, 3], 0, h)

        self._original_predictions = preds
        self._original_image_size = (h, w)

    def convert_original_predictions(self, **kwargs):
        """
        Convert raw predictions to SAHI ObjectPrediction format.

        Sets:
            self.object_prediction_list: list of ObjectPrediction
        """
        preds = self._original_predictions
        object_predictions = []

        for pred in preds:
            x1, y1, x2, y2 = pred[:4]
            scores = pred[4:]  # per-class scores
            max_cls = int(np.argmax(scores))
            max_score = float(scores[max_cls])

            if max_score < self._conf_thresh:
                continue

            # Filter tiny boxes
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            object_predictions.append(
                ObjectPrediction(
                    bbox=[x1, y1, x2, y2],
                    category_id=max_cls,
                    score=max_score,
                    category_name=self._class_names[max_cls],
                    bool_mask=None,
                )
            )

        self.object_prediction_list = object_predictions


def run_sahi_evaluation(model, img_dir, label_dir, device,
                        num_classes=2, class_names=None,
                        conf_thresh=0.25, iou_thresh=0.5,
                        slice_size=640, overlap_ratio=0.2,
                        verbose=True):
    """
    Run SAHI sliced evaluation on a dataset.

    Args:
        model: SAFE-Det model (eval mode)
        img_dir: path to images directory
        label_dir: path to YOLO labels directory
        device: torch device
        num_classes: number of classes
        class_names: list of class names
        conf_thresh: confidence threshold
        iou_thresh: IoU threshold for matching
        slice_size: size of each slice (pixels)
        overlap_ratio: overlap between adjacent slices (0-1)
        verbose: print progress

    Returns:
        dict with mAP50, per-class AP, precision, recall
    """
    check_sahi_available()
    import cv2
    from pathlib import Path

    if class_names is None:
        class_names = ['smoke', 'fire']

    # Wrap model for SAHI
    sahi_model = SAFEDetSahiModel(
        model=model, device=device,
        num_classes=num_classes, class_names=class_names,
        conf_thresh=conf_thresh
    )

    # Gather images
    img_dir = Path(img_dir)
    img_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ])

    if verbose:
        print(f"SAHI Evaluation: {len(img_files)} images")
        print(f"  Slice size: {slice_size}px, Overlap: {overlap_ratio:.0%}")

    all_detections = []  # (img_idx, cls, conf, x1, y1, x2, y2)
    all_targets = []     # (img_idx, cls, x1, y1, x2, y2)

    for idx, img_file in enumerate(img_files):
        if verbose and idx % 100 == 0:
            print(f"  Processing {idx}/{len(img_files)}...")

        # Run SAHI sliced prediction
        result = get_sliced_prediction(
            image=str(img_file),
            detection_model=sahi_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            perform_standard_pred=True,  # Also run on full image
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5,
            verbose=0,
        )

        # Collect predictions
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            all_detections.append(
                (idx, pred.category_id, pred.score.value,
                 bbox[0], bbox[1], bbox[2], bbox[3])
            )

        # Load ground truth (YOLO format)
        if label_dir:
            label_file = Path(label_dir) / (img_file.stem + '.txt')
            if label_file.exists():
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    with open(label_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:5])
                                x1 = (cx - bw / 2) * w
                                y1 = (cy - bh / 2) * h
                                x2 = (cx + bw / 2) * w
                                y2 = (cy + bh / 2) * h
                                all_targets.append((idx, cls, x1, y1, x2, y2))

    # Compute metrics (same logic as eval.py)
    results = _compute_metrics(all_detections, all_targets,
                               num_classes, class_names, iou_thresh)

    if verbose:
        print(f"\n{'='*60}")
        print(f"SAHI Results (slice={slice_size}, overlap={overlap_ratio})")
        print(f"{'='*60}")
        print(f"{'Class':<10} {'AP50':<8} {'P':<8} {'R':<8} {'GT':<6} {'Det':<6}")
        print("-" * 60)
        for name in class_names:
            if name in results:
                r = results[name]
                print(f"{name:<10} {r['AP50']:.4f}  {r.get('precision', 0):.4f}  "
                      f"{r.get('recall', 0):.4f}  {r.get('num_gt', 0):<6} {r.get('num_det', 0):<6}")
        print("-" * 60)
        print(f"{'mAP50':<10} {results['mAP50']:.4f}")
        print(f"{'='*60}")

    return results


def _compute_metrics(all_detections, all_targets, num_classes, class_names, iou_thresh):
    """Compute per-class AP50, precision, recall."""
    results = {}

    for cls in range(num_classes):
        cls_dets = [d for d in all_detections if d[1] == cls]
        cls_gts = [g for g in all_targets if g[1] == cls]

        if len(cls_gts) == 0:
            results[class_names[cls]] = {'AP50': 0.0, 'num_gt': 0, 'num_det': len(cls_dets)}
            continue

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
                iou = _compute_iou(det_box, gt_box)
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

        # AP (11-point or all-point interpolation)
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([1.], precision, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i_pts = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i_pts + 1] - mrec[i_pts]) * mpre[i_pts + 1])

        results[class_names[cls]] = {
            'AP50': ap,
            'precision': precision[-1] if len(precision) > 0 else 0,
            'recall': recall[-1] if len(recall) > 0 else 0,
            'num_gt': len(cls_gts),
            'num_det': len(cls_dets)
        }

    aps = [r['AP50'] for r in results.values() if isinstance(r, dict) and r.get('num_gt', 0) > 0]
    results['mAP50'] = np.mean(aps) if aps else 0.0
    return results


def _compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)
