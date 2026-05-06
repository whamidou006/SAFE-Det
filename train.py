"""
CCPE DDP Training Script for Fire/Smoke Detection.
Standalone implementation — no mmdetection dependency.

Usage:
    # Single GPU
    python train.py --config configs/ccpe_single_1024.yaml --gpu 0

    # Multi-GPU DDP
    torchrun --nproc_per_node=4 train.py --config configs/ccpe_single_1024.yaml
"""

import os
import sys
import yaml
import argparse
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from models import CCPE_Detector
from utils.dataset import FireSmokeDataset, collate_fn
from utils.assigner import SimOTAAssigner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='CCPE Fire/Smoke Training')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--gpu', type=int, default=None, help='Single GPU id (for non-DDP)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Evaluation only')
    return parser.parse_args()


def setup_distributed():
    """Initialize DDP if launched with torchrun."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


class YOLOXLoss:
    """YOLOX loss with SimOTA assignment and SNSM."""

    def __init__(self, num_classes, strides=(8, 16, 32), img_size=1024):
        self.num_classes = num_classes
        self.strides = strides
        self.img_size = img_size
        self.assigner = SimOTAAssigner(center_radius=2.5)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, cls_scores, bbox_preds, obj_scores, targets, model_head):
        """
        Compute YOLOX loss.

        Args:
            cls_scores: list of (B, C, H, W)
            bbox_preds: list of (B, 4, H, W)
            obj_scores: list of (B, 1, H, W)
            targets: list of (N_i, 5) tensors [cls, x1, y1, w, h]
        """
        device = cls_scores[0].device
        B = cls_scores[0].shape[0]

        # Generate all anchor points
        all_points = []
        all_strides = []
        for i, stride in enumerate(self.strides):
            H, W = cls_scores[i].shape[2], cls_scores[i].shape[3]
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device) * stride + stride / 2,
                torch.arange(W, device=device) * stride + stride / 2,
                indexing='ij'
            )
            points = torch.stack([xv.flatten(), yv.flatten()], dim=1)
            all_points.append(points)
            all_strides.append(torch.full((points.shape[0],), stride, device=device))

        points = torch.cat(all_points, dim=0)
        strides_flat = torch.cat(all_strides, dim=0)
        N = points.shape[0]

        # Flatten predictions
        flat_cls = torch.cat([s.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes) for s in cls_scores], 1)
        flat_bbox = torch.cat([b.permute(0, 2, 3, 1).reshape(B, -1, 4) for b in bbox_preds], 1)
        flat_obj = torch.cat([o.permute(0, 2, 3, 1).reshape(B, -1, 1) for o in obj_scores], 1)

        # Decode boxes
        pred_xy = points[None] + flat_bbox[..., :2] * strides_flat[None, :, None]
        pred_wh = flat_bbox[..., 2:].exp() * strides_flat[None, :, None]
        pred_x1y1 = pred_xy - pred_wh / 2
        pred_x2y2 = pred_xy + pred_wh / 2
        pred_boxes = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        total_cls_loss = 0
        total_bbox_loss = 0
        total_obj_loss = 0
        num_fg = 0

        for b in range(B):
            tgt = targets[b]
            if tgt.shape[0] == 0:
                # Negative image — only objectness loss
                obj_target = torch.zeros(N, 1, device=device)
                total_obj_loss += self.bce(flat_obj[b], obj_target).sum()
                continue

            # Convert targets from [cls, x1, y1, w, h] to [x1, y1, x2, y2]
            gt_labels = tgt[:, 0].long().to(device)
            gt_x1 = tgt[:, 1].to(device)
            gt_y1 = tgt[:, 2].to(device)
            gt_w = tgt[:, 3].to(device)
            gt_h = tgt[:, 4].to(device)
            gt_bboxes = torch.stack([gt_x1, gt_y1, gt_x1 + gt_w, gt_y1 + gt_h], dim=1)

            # SimOTA assignment
            with torch.no_grad():
                pred_obj_scores = flat_obj[b, :, 0].sigmoid()
                assigned_labels, assigned_bboxes, assigned_scores, fg_mask = \
                    self.assigner.assign(
                        pred_obj_scores, pred_boxes[b], gt_bboxes, gt_labels,
                        points, strides_flat, self.num_classes
                    )

            num_fg_this = fg_mask.sum().item()
            num_fg += num_fg_this

            # Objectness target
            obj_target = fg_mask.float().unsqueeze(-1)
            total_obj_loss += self.bce(flat_obj[b], obj_target).sum()

            if num_fg_this > 0:
                # Classification loss (only for positive anchors)
                cls_target = F.one_hot(assigned_labels[fg_mask], self.num_classes).float()
                cls_pred = flat_cls[b][fg_mask]
                total_cls_loss += self.bce(cls_pred, cls_target).sum()

                # IoU loss
                pred_fg = pred_boxes[b][fg_mask]
                gt_fg = assigned_bboxes[fg_mask]
                iou_loss = self._iou_loss(pred_fg, gt_fg)
                total_bbox_loss += iou_loss.sum()

        num_fg = max(num_fg, 1)
        loss_cls = total_cls_loss / num_fg
        loss_bbox = 5.0 * total_bbox_loss / num_fg
        loss_obj = total_obj_loss / N

        total_loss = loss_cls + loss_bbox + loss_obj
        return total_loss, {
            'loss_cls': loss_cls.item(),
            'loss_bbox': loss_bbox.item(),
            'loss_obj': loss_obj.item(),
            'num_fg': num_fg
        }

    def _iou_loss(self, pred, target):
        """IoU loss (1 - IoU)."""
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = area_pred + area_target - inter
        iou = inter / union.clamp(min=1e-6)
        return 1 - iou


def train_one_epoch(model, dataloader, optimizer, scaler, loss_fn, epoch, rank, cfg):
    model.train()
    total_loss = 0
    num_batches = 0
    log_interval = cfg.get('log_interval', 50)
    start_time = time.time()

    for batch_idx, (imgs, targets, _) in enumerate(dataloader):
        imgs = imgs.cuda(non_blocking=True)

        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            cls_scores, bbox_preds, obj_scores = model(imgs)
            loss, loss_dict = loss_fn(cls_scores, bbox_preds, obj_scores, targets,
                                      model.module.head if hasattr(model, 'module') else model.head)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if rank == 0 and (batch_idx % log_interval == 0):
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"loss={loss.item():.4f} cls={loss_dict['loss_cls']:.4f} "
                f"bbox={loss_dict['loss_bbox']:.4f} obj={loss_dict['loss_obj']:.4f} "
                f"fg={loss_dict['num_fg']} elapsed={elapsed:.1f}s"
            )

    avg_loss = total_loss / max(num_batches, 1)
    if rank == 0:
        logger.info(f"Epoch {epoch} complete — avg loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, loss_fn, rank):
    model.eval()
    total_loss = 0
    num_batches = 0

    for imgs, targets, _ in dataloader:
        imgs = imgs.cuda(non_blocking=True)
        with autocast(dtype=torch.bfloat16):
            cls_scores, bbox_preds, obj_scores = model(imgs)
            loss, _ = loss_fn(cls_scores, bbox_preds, obj_scores, targets, None)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    if rank == 0:
        logger.info(f"Validation loss: {avg_loss:.4f}")
    return avg_loss


def main():
    args = parse_args()
    cfg = load_config(args.config)

    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)

    # Output directory
    output_dir = Path(cfg.get('output_dir', 'runs/ccpe'))
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

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
        contrast_steps=model_cfg.get('contrast_steps'),
        use_checkpoint=model_cfg.get('use_checkpoint', False),
        pretrained_swin=model_cfg.get('pretrained_swin')
    )

    model.cuda()
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model params: {n_params / 1e6:.1f}M")

    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Dataset
    data_cfg = cfg['data']
    train_dataset = FireSmokeDataset(
        img_dir=data_cfg['train_img_dir'],
        ann_file=data_cfg.get('train_ann_file'),
        label_dir=data_cfg.get('train_label_dir'),
        img_size=model_cfg.get('input_size', [1024, 1024])[0],
        augment=True,
        mosaic_prob=data_cfg.get('mosaic_prob', 1.0),
        mixup_prob=data_cfg.get('mixup_prob', 0.5),
        num_classes=model_cfg['num_classes']
    )

    val_dataset = FireSmokeDataset(
        img_dir=data_cfg['val_img_dir'],
        ann_file=data_cfg.get('val_ann_file'),
        label_dir=data_cfg.get('val_label_dir'),
        img_size=model_cfg.get('input_size', [1024, 1024])[0],
        augment=False,
        num_classes=model_cfg['num_classes']
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset, batch_size=data_cfg.get('batch_size', 4),
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=data_cfg.get('num_workers', 8),
        collate_fn=collate_fn, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=data_cfg.get('batch_size', 4),
        sampler=val_sampler, shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        collate_fn=collate_fn, pin_memory=True
    )

    if is_main:
        logger.info(f"Train: {len(train_dataset)} images, {len(train_loader)} batches/epoch")
        logger.info(f"Val: {len(val_dataset)} images")

    # Optimizer
    opt_cfg = cfg.get('optimizer', {})
    lr = opt_cfg.get('lr', 0.0001)
    weight_decay = opt_cfg.get('weight_decay', 5e-4)

    param_groups = (model.module if hasattr(model, 'module') else model).get_param_groups(lr, weight_decay)
    optimizer = torch.optim.SGD(
        param_groups,
        momentum=opt_cfg.get('momentum', 0.9),
        nesterov=True
    )

    # Scheduler
    train_cfg = cfg.get('training', {})
    max_epochs = train_cfg.get('max_epochs', 80)
    warmup_epochs = train_cfg.get('warmup_epochs', 2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs, eta_min=lr * 0.05
    )

    # Loss
    loss_fn = YOLOXLoss(
        num_classes=model_cfg['num_classes'],
        img_size=model_cfg.get('input_size', [1024, 1024])[0]
    )

    # AMP scaler
    scaler = GradScaler()

    # Resume
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_state = ckpt['model_state_dict']
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        if is_main:
            logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # Warmup LR
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * warmup_factor

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, epoch, rank, cfg)

        if epoch >= warmup_epochs:
            scheduler.step()

        # Validation
        if (epoch + 1) % train_cfg.get('val_interval', 5) == 0:
            val_loss = validate(model, val_loader, loss_fn, rank)

            if is_main and val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, epoch, best_loss, output_dir / 'best.pth')
                logger.info(f"New best model saved (loss={best_loss:.4f})")

        # Save periodic checkpoint
        if is_main and (epoch + 1) % train_cfg.get('save_interval', 10) == 0:
            save_checkpoint(model, optimizer, epoch, best_loss, output_dir / f'epoch_{epoch}.pth')

    # Final save
    if is_main:
        save_checkpoint(model, optimizer, max_epochs - 1, best_loss, output_dir / 'final.pth')
        logger.info("Training complete!")

    if world_size > 1:
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, epoch, best_loss, path):
    state = {
        'epoch': epoch,
        'model_state_dict': (model.module if hasattr(model, 'module') else model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }
    torch.save(state, path)


if __name__ == '__main__':
    main()
