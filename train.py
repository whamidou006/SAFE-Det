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

from models import CCPE_Detector, FireSightDetector
from utils.dataset import FireSmokeDataset, collate_fn
from utils.assigner import SimOTAAssigner, build_assigner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def build_model(cfg: dict):
    """Instantiate either CCPE_Detector or FireSightDetector based on
    ``cfg['model']['type']`` (defaults to 'ccpe' for back-compat).

    Raises ValueError on an unknown type so silently-mis-configured runs
    fail loudly.
    """
    model_cfg = cfg['model']
    model_type = model_cfg.get('type', 'ccpe').lower()

    if model_type == 'ccpe':
        return CCPE_Detector(
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
            pretrained_swin=model_cfg.get('pretrained_swin'),
        )
    if model_type == 'firesight':
        return FireSightDetector(
            num_classes=model_cfg['num_classes'],
            backbone_type=model_cfg.get('backbone_type', 'dinov2'),
            backbone_channels=model_cfg.get('backbone_channels', 256),
            use_dcm=model_cfg.get('use_dcm', True),
            use_fam=model_cfg.get('use_fam', True),
            use_tm=model_cfg.get('use_tm', True),
            use_temporal=model_cfg.get('use_temporal', False),
            head_type=model_cfg.get('head_type', 'yolox'),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
            input_size=tuple(model_cfg.get('input_size', [1024, 1024])),
        )
    raise ValueError(
        f"Unknown model.type {model_type!r} — expected 'ccpe' or 'firesight'."
    )


def parse_args():
    parser = argparse.ArgumentParser(description='CCPE Fire/Smoke Training')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--gpu', type=int, default=None, help='Single GPU id (for non-DDP)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Evaluation only')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override per-GPU batch size from the YAML '
                             '(useful for OOM fallbacks without editing config).')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override base learning rate from the YAML '
                             '(typically combined with --batch-size).')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override training.max_epochs from the YAML.')
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
    """YOLOX loss with SimOTA assignment and SNSM.

    The bbox regression term is dispatched through `models.losses_nwd.bbox_loss`
    so it can optionally use NWD or a CIoU/NWD mix instead of plain IoU.
    Default behaviour is unchanged (`bbox_loss_type='ciou'`).
    """

    def __init__(self, num_classes, strides=(8, 16, 32), img_size=1024,
                 bbox_loss_type='ciou', nwd_constant=12.8, nwd_mix_weight=0.5,
                 assigner='simota', assigner_kwargs=None):
        self.num_classes = num_classes
        self.strides = strides
        self.img_size = img_size
        self.assigner = build_assigner(assigner, **(assigner_kwargs or {}))
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # bbox-regression loss config (see models/losses_nwd.py).
        self.bbox_loss_type = bbox_loss_type
        self.nwd_constant = nwd_constant
        self.nwd_mix_weight = nwd_mix_weight

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

        # Initialise as zero-tensors so accumulation stays in tensor-land even
        # when no positive sample exists in any image (purely-negative batch).
        zero = flat_cls.new_zeros(())
        total_cls_loss = zero.clone()
        total_bbox_loss = zero.clone()
        total_obj_loss = zero.clone()
        num_fg = 0

        for b in range(B):
            tgt = targets[b]
            if tgt.shape[0] == 0:
                # Negative image — only objectness loss
                obj_target = torch.zeros(N, 1, device=device)
                total_obj_loss = total_obj_loss + self.bce(flat_obj[b], obj_target).sum()
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
            total_obj_loss = total_obj_loss + self.bce(flat_obj[b], obj_target).sum()

            if num_fg_this > 0:
                # Classification loss (only for positive anchors)
                cls_target = F.one_hot(assigned_labels[fg_mask], self.num_classes).float()
                cls_pred = flat_cls[b][fg_mask]
                total_cls_loss = total_cls_loss + self.bce(cls_pred, cls_target).sum()

                # IoU loss
                pred_fg = pred_boxes[b][fg_mask]
                gt_fg = assigned_bboxes[fg_mask]
                iou_loss = self._iou_loss(pred_fg, gt_fg)
                total_bbox_loss = total_bbox_loss + iou_loss.sum()

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
        """Per-box bbox regression loss. Dispatches to NWD or CIoU/NWD-mix
        when the corresponding config flag is set; otherwise plain 1-IoU.
        Kept under the same name to preserve the original call site
        unchanged."""
        from models.losses_nwd import bbox_loss
        return bbox_loss(
            pred, target,
            loss_type=self.bbox_loss_type,
            nwd_constant=self.nwd_constant,
            nwd_mix_weight=self.nwd_mix_weight,
        )


def train_one_epoch(model, dataloader, optimizer, scaler, loss_fn, epoch, rank, cfg):
    model.train()
    total_loss = 0
    num_batches = 0
    log_interval = cfg.get('log_interval', 50)
    start_time = time.time()

    base = model.module if hasattr(model, 'module') else model
    head_type = getattr(base, 'head_type', 'yolox')
    img_size = cfg['model'].get('input_size', [1024, 1024])[0]

    for batch_idx, (imgs, targets, _) in enumerate(dataloader):
        imgs = imgs.cuda(non_blocking=True)

        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            if head_type == 'dfine':
                # Pass targets into the model so the contrastive
                # denoising group can be built inside DFINETransformer.
                from models.firesight.dfine_runtime import (
                    convert_yolox_targets_to_dfine,
                )
                dfine_targets = convert_yolox_targets_to_dfine(
                    targets, img_size=img_size, device=imgs.device,
                )
                outputs = model(imgs, targets=dfine_targets)
                loss, loss_dict = loss_fn(outputs, targets, img_size)
            else:
                cls_scores, bbox_preds, obj_scores = model(imgs)
                loss, loss_dict = loss_fn(
                    cls_scores, bbox_preds, obj_scores, targets,
                    base.head,
                )

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
                f"loss={loss.item():.4f} cls={loss_dict.get('loss_cls', 0.0):.4f} "
                f"bbox={loss_dict.get('loss_bbox', 0.0):.4f} "
                f"obj={loss_dict.get('loss_obj', 0.0):.4f} "
                f"fg={loss_dict.get('num_fg', 0)} elapsed={elapsed:.1f}s"
            )

    avg_loss = total_loss / max(num_batches, 1)
    if rank == 0:
        logger.info(f"Epoch {epoch} complete — avg loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, loss_fn, rank, cfg):
    """Compute validation loss. Dispatches on the model's head_type so the
    same function works for the YOLOX path and the D-FINE / DETR path
    (which returns a dict instead of a 3-tuple from forward())."""
    model.eval()
    total_loss = 0
    num_batches = 0

    base = model.module if hasattr(model, 'module') else model
    head_type = getattr(base, 'head_type', 'yolox')
    img_size = cfg['model'].get('input_size', [1024, 1024])[0]

    for imgs, targets, _ in dataloader:
        imgs = imgs.cuda(non_blocking=True)
        with autocast(dtype=torch.bfloat16):
            if head_type == 'dfine':
                from models.firesight.dfine_runtime import (
                    convert_yolox_targets_to_dfine,
                )
                dfine_targets = convert_yolox_targets_to_dfine(
                    targets, img_size=img_size, device=imgs.device,
                )
                outputs = model(imgs, targets=dfine_targets)
                loss, _ = loss_fn(outputs, targets, img_size)
            else:
                cls_scores, bbox_preds, obj_scores = model(imgs)
                loss, _ = loss_fn(cls_scores, bbox_preds, obj_scores,
                                  targets, base.head)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    if rank == 0:
        logger.info(f"Validation loss: {avg_loss:.4f}")
    return avg_loss


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides — applied before anything reads the config so the
    # whole pipeline (dataset, optimizer, scheduler, log lines) sees
    # the new values consistently.
    if args.batch_size is not None:
        cfg.setdefault('data', {})['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg.setdefault('optimizer', {})['lr'] = args.lr
    if args.max_epochs is not None:
        cfg.setdefault('training', {})['max_epochs'] = args.max_epochs

    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)

    # Output directory
    output_dir = Path(cfg.get('output_dir', 'runs/ccpe'))
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model_cfg = cfg['model']
    model = build_model(cfg)

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
    base_lrs = []
    weight_decay = opt_cfg.get('weight_decay', 5e-4)

    # FireSightDetector does not have get_param_groups; build one default group.
    base = model.module if hasattr(model, 'module') else model
    if hasattr(base, 'get_param_groups'):
        param_groups = base.get_param_groups(lr, weight_decay)
    else:
        param_groups = [{'params': [p for p in base.parameters() if p.requires_grad],
                         'lr': lr, 'weight_decay': weight_decay}]

    opt_name = opt_cfg.get('name', 'sgd').lower()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups, momentum=opt_cfg.get('momentum', 0.9), nesterov=True
        )
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise ValueError(f"Unknown optimizer.name {opt_name!r}")

    # Cache base LR per param group so warmup can multiply *that* (not the
    # already-warmed-up LR, which compounds).
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    # Scheduler
    train_cfg = cfg.get('training', {})
    max_epochs = train_cfg.get('max_epochs', 80)
    warmup_epochs = train_cfg.get('warmup_epochs', 2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs, eta_min=lr * 0.05
    )

    # Loss
    loss_cfg = cfg.get('loss', {})
    base_for_head = model.module if hasattr(model, 'module') else model
    head_type = getattr(base_for_head, 'head_type', 'yolox')
    if head_type == 'dfine':
        # D-FINE uses a Hungarian-matching DETR-style criterion that is
        # incompatible with YOLOXLoss's API. Build the wrapper from
        # dfine_runtime, which converts targets and exposes the same
        # (loss, log_dict) tuple train_one_epoch expects.
        from models.firesight.dfine_runtime import build_dfine_criterion
        model_node = cfg['model']
        loss_fn = build_dfine_criterion(
            num_classes=model_cfg['num_classes'],
            source=model_node.get('dfine_source'),
            weight_dict=loss_cfg.get('dfine_weight_dict'),
            losses=loss_cfg.get('dfine_losses'),
            reg_max=loss_cfg.get('dfine_reg_max', 32),
        )
    else:
        loss_fn = YOLOXLoss(
            num_classes=model_cfg['num_classes'],
            img_size=model_cfg.get('input_size', [1024, 1024])[0],
            bbox_loss_type=loss_cfg.get('bbox_loss_type', 'ciou'),
            nwd_constant=loss_cfg.get('nwd_constant', 12.8),
            nwd_mix_weight=loss_cfg.get('nwd_mix_weight', 0.5),
            assigner=loss_cfg.get('assigner', 'simota'),
            assigner_kwargs=loss_cfg.get('assigner_kwargs', None),
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

        # Warmup LR: linearly ramp from 1/warmup_epochs of base LR up to base.
        # Use cached base_lrs so we don't compound the multiplier each epoch.
        if epoch < warmup_epochs and warmup_epochs > 0:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg['lr'] = base_lr * warmup_factor
        elif epoch == warmup_epochs:
            # Reset to base LR exactly when warmup completes.
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg['lr'] = base_lr

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, epoch, rank, cfg)

        if epoch >= warmup_epochs:
            scheduler.step()

        # Validation
        if (epoch + 1) % train_cfg.get('val_interval', 5) == 0:
            val_loss = validate(model, val_loader, loss_fn, rank, cfg)

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
