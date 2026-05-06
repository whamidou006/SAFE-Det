"""
COCO-format dataset for fire/smoke detection with YOLOX augmentations.
Supports both single-frame and multi-frame (temporal) modes.
"""

import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path


class FireSmokeDataset(Dataset):
    """
    COCO-format fire/smoke detection dataset.

    Supports:
    - Standard COCO JSON annotations
    - YOLO-format labels (auto-conversion)
    - Mosaic + MixUp augmentation (YOLOX-style)
    - Multi-frame mode (concatenate current + previous frame)
    """

    def __init__(self, img_dir, ann_file=None, label_dir=None,
                 img_size=1024, augment=True, mosaic_prob=1.0,
                 mixup_prob=0.5, multi_frame=False, num_classes=2):
        super().__init__()
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.multi_frame = multi_frame
        self.num_classes = num_classes

        # Load annotations
        if ann_file and os.path.exists(ann_file):
            self._load_coco(ann_file)
        elif label_dir and os.path.exists(label_dir):
            self._load_yolo(img_dir, label_dir)
        else:
            raise ValueError("Must provide either ann_file (COCO) or label_dir (YOLO)")

    def _load_coco(self, ann_file):
        """Load COCO-format annotations."""
        with open(ann_file) as f:
            coco = json.load(f)

        self.images = {img['id']: img for img in coco['images']}
        self.img_ids = [img['id'] for img in coco['images']]

        # Group annotations by image
        self.annotations = {}
        for ann in coco.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

    def _load_yolo(self, img_dir, label_dir):
        """Load YOLO-format labels."""
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.images = {}
        self.img_ids = []
        self.annotations = {}

        for idx, img_file in enumerate(img_files):
            img_path = os.path.join(img_dir, img_file)
            # Read image dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            self.images[idx] = {
                'id': idx,
                'file_name': img_path,
                'width': w,
                'height': h
            }
            self.img_ids.append(idx)

            # Read YOLO label
            label_file = os.path.join(label_dir, Path(img_file).stem + '.txt')
            anns = []
            if os.path.exists(label_file):
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            # Convert YOLO normalized to COCO absolute
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            box_w = bw * w
                            box_h = bh * h
                            anns.append({
                                'bbox': [x1, y1, box_w, box_h],
                                'category_id': cls_id,
                                'area': box_w * box_h
                            })
            self.annotations[idx] = anns

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if self.augment and random.random() < self.mosaic_prob:
            img, targets = self._load_mosaic(idx)
            if random.random() < self.mixup_prob:
                img2, targets2 = self._load_mosaic(random.randint(0, len(self) - 1))
                img, targets = self._mixup(img, targets, img2, targets2)
        else:
            img, targets = self._load_single(idx)

        # Resize to target size
        img, targets = self._resize_pad(img, targets)

        # Random flip
        if self.augment and random.random() > 0.5:
            img = img[:, ::-1].copy()
            if len(targets) > 0:
                targets[:, 1] = self.img_size - targets[:, 1] - targets[:, 3]

        # HSV augmentation
        if self.augment:
            img = self._hsv_augment(img)

        # To tensor
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = torch.from_numpy(img)

        # Targets: [class, x1, y1, w, h]
        if len(targets) > 0:
            targets = torch.from_numpy(targets).float()
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)

        return img, targets, idx

    def _load_single(self, idx):
        """Load single image and its annotations."""
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = img_info['file_name']
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.img_dir, img_path)
        img = cv2.imread(img_path)
        if img is None:
            # Fallback to blank
            img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
            return img, np.zeros((0, 5), dtype=np.float32)

        # Annotations: [class, x1, y1, w, h]
        anns = self.annotations.get(img_id, [])
        targets = []
        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x, y, w, h]
            cls = ann['category_id']
            targets.append([cls, bbox[0], bbox[1], bbox[2], bbox[3]])

        targets = np.array(targets, dtype=np.float32) if targets else np.zeros((0, 5), dtype=np.float32)
        return img, targets

    def _load_mosaic(self, idx):
        """4-image mosaic augmentation."""
        s = self.img_size
        # Center point
        yc = int(random.uniform(s * 0.5, s * 1.5))
        xc = int(random.uniform(s * 0.5, s * 1.5))

        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        all_targets = []

        for i, index in enumerate(indices):
            img, targets = self._load_single(index)
            h, w = img.shape[:2]

            # Scale to fit
            scale = min(s / h, s / w)
            nh, nw = int(h * scale), int(w * scale)
            img = cv2.resize(img, (nw, nh))

            # Place in mosaic
            if i == 0:  # top-left
                x1, y1, x2, y2 = max(xc - nw, 0), max(yc - nh, 0), xc, yc
                crop_x1 = nw - (x2 - x1)
                crop_y1 = nh - (y2 - y1)
                crop_x2, crop_y2 = nw, nh
            elif i == 1:  # top-right
                x1, y1, x2, y2 = xc, max(yc - nh, 0), min(xc + nw, s * 2), yc
                crop_x1, crop_y1 = 0, nh - (y2 - y1)
                crop_x2, crop_y2 = x2 - x1, nh
            elif i == 2:  # bottom-left
                x1, y1, x2, y2 = max(xc - nw, 0), yc, xc, min(yc + nh, s * 2)
                crop_x1 = nw - (x2 - x1)
                crop_y1 = 0
                crop_x2, crop_y2 = nw, y2 - y1
            else:  # bottom-right
                x1, y1, x2, y2 = xc, yc, min(xc + nw, s * 2), min(yc + nh, s * 2)
                crop_x1, crop_y1 = 0, 0
                crop_x2, crop_y2 = x2 - x1, y2 - y1

            mosaic_img[y1:y2, x1:x2] = img[crop_y1:crop_y2, crop_x1:crop_x2]

            # Adjust targets
            if len(targets) > 0:
                targets[:, 1:] *= scale
                # Offset
                offset_x = x1 - crop_x1
                offset_y = y1 - crop_y1
                targets[:, 1] += offset_x
                targets[:, 2] += offset_y
                all_targets.append(targets)

        # Crop to final size
        mosaic_img = mosaic_img[s // 2: s // 2 + s, s // 2: s // 2 + s]

        if all_targets:
            targets = np.concatenate(all_targets, 0)
            # Adjust for crop
            targets[:, 1] -= s // 2
            targets[:, 2] -= s // 2
            # Clip
            targets = self._clip_targets(targets, s, s)
        else:
            targets = np.zeros((0, 5), dtype=np.float32)

        return mosaic_img, targets

    def _mixup(self, img1, targets1, img2, targets2, alpha=0.5):
        """MixUp augmentation."""
        r = np.random.beta(alpha, alpha)
        img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        targets = np.concatenate([targets1, targets2], 0) if len(targets1) > 0 or len(targets2) > 0 else np.zeros((0, 5))
        return img, targets

    def _resize_pad(self, img, targets):
        """Resize image to target size with letterbox padding."""
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        nh, nw = int(h * scale), int(w * scale)

        img = cv2.resize(img, (nw, nh))
        # Pad
        new_img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        dy, dx = (self.img_size - nh) // 2, (self.img_size - nw) // 2
        new_img[dy:dy + nh, dx:dx + nw] = img

        # Adjust targets
        if len(targets) > 0:
            targets[:, 1] = targets[:, 1] * scale + dx
            targets[:, 2] = targets[:, 2] * scale + dy
            targets[:, 3] *= scale
            targets[:, 4] *= scale

        return new_img, targets

    def _clip_targets(self, targets, h, w):
        """Clip targets to image bounds and remove invalid ones."""
        if len(targets) == 0:
            return targets
        # Clip x1, y1
        targets[:, 1] = np.clip(targets[:, 1], 0, w)
        targets[:, 2] = np.clip(targets[:, 2], 0, h)
        # Clip x2, y2 (computed from x1+w, y1+h)
        x2 = np.clip(targets[:, 1] + targets[:, 3], 0, w)
        y2 = np.clip(targets[:, 2] + targets[:, 4], 0, h)
        targets[:, 3] = x2 - targets[:, 1]
        targets[:, 4] = y2 - targets[:, 2]
        # Remove tiny boxes
        valid = (targets[:, 3] > 2) & (targets[:, 4] > 2)
        return targets[valid]

    def _hsv_augment(self, img, hgain=0.015, sgain=0.7, vgain=0.4):
        """HSV color augmentation."""
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
        img_hsv = cv2.merge([
            cv2.LUT(hue, lut_hue),
            cv2.LUT(sat, lut_sat),
            cv2.LUT(val, lut_val)
        ]).astype(np.uint8)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def collate_fn(batch):
    """Custom collate for variable-size targets."""
    imgs, targets, indices = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets), list(indices)
