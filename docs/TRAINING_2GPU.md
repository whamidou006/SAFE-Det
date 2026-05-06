# Training on 2 GPUs — runbook

All commands assume:

```bash
conda activate /home/whamidouche/ssdprivate/conda_envs/condor-bench
cd /home/whamidouche/ssdprivate/SAFE-Det
pytest -q                                    # must show 67 passed
```

Replace `0,1` below with whichever two GPU IDs you want to use
(`nvidia-smi` to list them).

---

## 0. One-time: Swin-T ImageNet pretrained weights

The two CCPE Swin-Tiny configs (`ccpe_single_1024.yaml` and
`ccpe_multi_1024.yaml`) load ImageNet-pretrained transformer weights
from `checkpoints/swin_tiny_patch4_window7_224.pth` (~115 MB). If the
file is missing, download it once:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/swin_tiny_patch4_window7_224.pth \
  https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
```

`load_pretrained()` fills 177/233 backbone keys (every Swin transformer
block, attention table and stage norm). The remaining 56 keys are the
CCPE patch_embed + contrast modules, which intentionally train from
scratch since the official Swin checkpoint has no equivalent.

For `ccpe_base_1024.yaml` (Swin-Base), download the matching
`swin_base_patch4_window7_224.pth` from the same release and uncomment
the `pretrained_swin` line in the config.

---

## 1. Per-config recipe (sized for 2× H100 NVL 95 GB)

`batch_size` in the table is **per-GPU** (what the YAML carries).
**Global batch** = `per-GPU batch × 2 GPUs`. LR has already been
linear-scaled (Goyal et al. 2017) for the new batch and warmup is
extended to 5 epochs to absorb the higher initial learning rate.

| # | Config                                | per-GPU batch | Global batch (2 GPUs) | Img size | LR (config) | VRAM/GPU* | Epochs |
|---|---------------------------------------|--------------:|----------------------:|---------:|------------:|----------:|-------:|
| 1 | `ccpe_single_1024.yaml`               | 32 | 64 | 1024 | 8e-4 (SGD)†† | ~30 GB | 50 |
| 2 | `ccpe_multi_1024.yaml`                | 32 | 64 | 1024 | 8e-4 (SGD)†† | ~40 GB | 50 |
| 3 | `ccpe_base_1024.yaml`                 | 32 | 64 | 1024 | 8e-4 (SGD)   | ~50 GB | 50 |
| 4 | `firesight_s_1024.yaml`               | 32 | 64 | 1022 | 1.6e-3 (SGD) | ~90 GB† | 50 |
| 5 | `firesight_st_1024.yaml`              | 32 | 64 | 1022 | 1.28e-3 (SGD)| OOM† | 50 |
| 6 | `firesight_s_nwd_tal_1024.yaml`       | 32 | 64 | 1022 | 1.6e-3 (SGD) | ~90 GB† | 50 |
| 7 | `firesight_dfine_1024.yaml`           | 32 | 64 | 1022 | 1.6e-3 (AdamW)| OOM† | 50 |

> *Estimates at bf16 autocast with `use_checkpoint: true` on CCPE
> configs (re-enabled — see below) and DINOv2 unfrozen. Numbers can
> drift ±15% with different mosaic crops. `scripts/train_ddp.sh`
> exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to
> further reduce fragmentation. **If you still OOM**, set
> `batch_size: 16` (or 8) — the linear-scaling LR for batch 16 is
> half: 4e-4 (CCPE) / 8e-4 (FireSight).
>
> **Note on `use_checkpoint`:** initially set to `false` on CCPE
> configs to save ~30% compute, but bs=32 / 1024² activations peak
> > 90 GB on busy mosaic batches and the run OOMed after a few
> epochs even with `expandable_segments:True`. Gradient checkpointing
> recomputes Swin activations during backward — the peak drops ~3×
> (to ~30 GB on `ccpe_single`), trading ~30% wall-clock speed for the
> headroom needed to actually finish training. If you have a 140 GB+
> H200 you can flip it back to `false`.
>
> †FireSight at batch 32 may not fit on H100 NVL (95 GB) because
> DINOv2 activations are not checkpointed. Recommended fallback for
> FireSight is `batch_size: 16, lr: 8e-4` (or `lr: 6.4e-4` for `_st`,
> `lr: 8e-4` for D-FINE / AdamW).
>
> All configs train for **50 epochs**, validate **every epoch**
> (`val_interval: 1`) and snapshot **every 2 epochs**
> (`save_interval: 2`). The best-loss model is saved as
> `runs/<config>/best.pth` and overwritten whenever val loss improves
> (val loss is averaged across DDP ranks for fair selection).
>
> ††**Pretrained Swin is currently DISABLED** (`pretrained_swin: null`)
> on all CCPE configs because the loaded ImageNet weights interact
> badly with bf16 autocast — symptom is a 96% GradScaler skip rate
> within 50 batches and a frozen loss curve. The
> `checkpoints/swin_tiny_patch4_window7_224.pth` file is downloaded
> and ready, and the loader maps 177/233 keys cleanly; re-enabling
> requires casting the Swin attention path (QKV matmul + softmax +
> relative_position_bias add) to fp32 inside `models/swin_ccpe.py`.
> Until then, the CCPE configs train Swin from random init — the
> previously-known-good path. `ccpe_base_1024.yaml` is unaffected
> (no pretrained weights downloaded for Swin-Base yet).
>
> The cosine LR scheduler uses **per-group `eta_min`** so the backbone
> floors at `backbone_lr * 0.05` and the head at `head_lr * 0.05`,
> preserving the 10:1 ratio throughout annealing. Override with
> `training.eta_min_ratio` in YAML.
>
> `num_workers` is set to 8 (was 16; lowered to fit `/dev/shm = 16 GB`).
> Under bf16 autocast `GradScaler` is **kept enabled** because its
> `step()` call detects inf/NaN gradients and silently skips the
> optimizer step — without that safety net a single bad batch
> permanently NaNs the model. The actual loss-scaling part is a no-op
> for bf16 (scale stays ~1.0). A defensive non-finite-loss check also
> short-circuits the step before backward in the same code path.

---

## 2. Commands — one per config (2-GPU DDP)

```bash
# 1. CCPE single-frame baseline
bash scripts/train_ddp.sh configs/ccpe_single_1024.yaml          0,1

# 2. CCPE multi-frame (6-channel concat)
bash scripts/train_ddp.sh configs/ccpe_multi_1024.yaml           0,1

# 3. CCPE Swin-B (larger backbone)
bash scripts/train_ddp.sh configs/ccpe_base_1024.yaml            0,1

# 4. FireSight-S (DINOv2-S + DCM/FAM/TM + YOLOX)
bash scripts/train_ddp.sh configs/firesight_s_1024.yaml          0,1

# 5. FireSight-ST (FireSight-S + Temporal Motion Fusion)
bash scripts/train_ddp.sh configs/firesight_st_1024.yaml         0,1

# 6. FireSight-S + NWD loss + TAL assigner (small-smoke ablation)
bash scripts/train_ddp.sh configs/firesight_s_nwd_tal_1024.yaml  0,1

# 7. FireSight + D-FINE decoder head (vendored rtv4 — no extra checkout)
bash scripts/train_ddp.sh configs/firesight_dfine_1024.yaml      0,1
```

### 2.1. Override batch / LR / epochs without editing the YAML

`train_ddp.sh` forwards any extra args after `<gpus>` straight to
`train.py`. The most common knobs are:

```bash
# Halve the batch (and LR) if you OOM at 32:
bash scripts/train_ddp.sh configs/firesight_s_1024.yaml 0,1 \
     --batch-size 16 --lr 8e-4

# Quick smoke run (1 epoch):
bash scripts/train_ddp.sh configs/ccpe_single_1024.yaml 0,1 --max-epochs 1
```

The CLI flags override the YAML *before* dataset/optimizer/scheduler
construction so every component sees consistent values.

Each run writes to `runs/<config-stem>/{best.pth, epoch_*.pth, final.pth}`.

---

## 3. Linear LR scaling — when to adjust

The configs were sized assuming the **4-GPU** DDP recipe with
per-GPU batch as listed (so global batch is doubled vs. the table
above when running on 4 GPUs). On 2 GPUs the global batch is halved,
which by the linear-scaling rule means LR should be halved too:

| Config family   | LR @ 4 GPUs (default) | LR @ 2 GPUs (suggested) |
|-----------------|----------------------:|------------------------:|
| CCPE *          | 1e-4                  | 5e-5                    |
| FireSight-S/-ST | 1e-4 / 8e-5           | 5e-5 / 4e-5             |
| FireSight-DFINE | 1e-4 (AdamW)          | 5e-5 (AdamW)            |

Apply by editing the config in place, e.g.:

```yaml
optimizer:
  lr: 0.00005          # was 0.0001
```

Or keep the LR and **double `warmup_epochs`** instead — this is the
"scale by warmup" alternative that often works better for transformer
backbones.

> If you'd rather keep the global batch identical to the 4-GPU recipe,
> double the per-GPU `batch_size` (only feasible on 80 GB cards for the
> CCPE family; FireSight is already memory-bound).

---

## 4. Recommended batch sizes for other GPU sizes

| GPU                | CCPE (per-GPU) | FireSight-S/-ST | FireSight-D-FINE |
|--------------------|---------------:|----------------:|-----------------:|
| A100 80 GB         | 4              | 2               | 2                |
| A100 40 GB / A40   | 2              | 2 (+ checkpoint)| 1 (+ checkpoint) |
| RTX 6000 Ada 48 GB | 2              | 2               | 1 — 2            |
| L40 / A6000 48 GB  | 2              | 2               | 1                |
| V100 32 GB         | 2 (+ checkpoint) | 1 (+ checkpoint) | not recommended |

To enable gradient checkpointing on FireSight backbones, add to the
`model:` block:

```yaml
freeze_backbone: false
backbone_kwargs: { use_checkpoint: true }   # if your backbone supports it
```

For CCPE configs `use_checkpoint: true` is already set in
`ccpe_single_1024.yaml` and inherited by the others.

---

## 5. Resume, eval, and tensorboard

```bash
# Resume from epoch_29
bash scripts/train_ddp.sh configs/firesight_s_1024.yaml 0,1 \
     --resume runs/firesight_s_1024/epoch_29.pth   # (pass-through arg)

# Single-GPU evaluation on either val (random) or test (disjoint cameras)
python eval.py --config configs/firesight_dfine_1024.yaml \
               --checkpoint runs/firesight_dfine_1024/best.pth \
               --split test --gpu 0
```

---

## 6. Recommended training order

If you want to stage the campaign rather than launch all 7 in parallel:

1. **#1 ccpe_single_1024** — establishes a working YOLOX baseline + verifies
   the dataset path / dataloader on this machine. Smallest model, fastest
   to converge — finishes first.
2. **#4 firesight_s_1024** — gets the DINOv2 baseline. Compare AP50 vs. #1
   to confirm the SAFE modules are helping.
3. **#7 firesight_dfine_1024** — production-parity comparison; uses the
   vendored rtv4 decoder + Hungarian criterion. Needs the most VRAM.
4. **#6 firesight_s_nwd_tal_1024** — small-smoke / tiny-object ablation.
   Run this only after you have a #4 baseline to compare against.
5. **#5 firesight_st_1024** — only meaningful if your dataset has the
   per-camera frame-pair sequences (`frame_interval: 60` in the config).
6. **#2 ccpe_multi_1024**, **#3 ccpe_base_1024** — larger / multi-frame
   CCPE variants for completeness.
