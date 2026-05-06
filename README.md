# SAFE-Det — Smoke-Aware Feature Enhancement Detector

Novel fire/smoke detection architecture for wildfire early warning systems.
Combines ideas from CCPE (Wang et al., IJIS 2025) with modern vision transformers
and domain-specific innovations.

## Architectures

### CCPE Baseline (reimplemented)
- **Backbone**: Swin Transformer + Cross Contrast Patch Embedding (CCPE)
- **Neck**: YOLOX-style PAFPN
- **Head**: YOLOX + Separable Negative Sampling Mechanism (SNSM)
- **Key idea**: Multi-scale spatial contrast (1-128px shifts) captures soft smoke edges

### FireSight (novel — our contribution)
- **Backbone**: DINOv2 ViT (self-supervised, much stronger features)
- **SAFE modules**:
  - **DCM** — Deformable Contrast Module: learned spatial offsets (replaces fixed shifts)
  - **FAM** — Frequency Attention Module: frequency-domain channel reweighting
  - **TM** — Transparency Module: feature-level background subtraction for semi-transparent smoke
  - **Temporal Motion Fusion**: motion-guided cross-frame attention
- **Head**: YOLOX + SNSM (or DETR decoder)

## Model Variants

| Config | Backbone | SAFE | Temporal | Params | Description |
|--------|----------|------|----------|--------|-------------|
| `ccpe_single_1024` | Swin-T + CCPE | — | No | ~30M | CCPE baseline, single-frame |
| `ccpe_multi_1024` | Swin-T + CCPE | — | 6ch concat | ~30M | CCPE multi-frame |
| `ccpe_base_1024` | Swin-B + CCPE | — | No | ~88M | CCPE larger backbone |
| `firesight_s_1024` | DINOv2-S | DCM+FAM+TM | No | ~35M | FireSight single-frame |
| `firesight_st_1024` | DINOv2-S | DCM+FAM+TM | Yes | ~40M | FireSight with temporal |

## Dataset

ALERTCalifornia wildfire detection dataset (YOLO format):
- **Train**: 178,394 images, 2 classes (smoke=0, fire=1)
- **Val**: 9,888 images (same cameras as train)
- **Test**: 4,967 images (40 unseen cameras — generalization test)

## Usage

### Training

```bash
# Single-GPU
python train.py --config configs/ccpe_single_1024.yaml --gpu 0

# Multi-GPU DDP (4 GPUs: 0,1,2,3)
bash scripts/train_ddp.sh configs/ccpe_single_1024.yaml 0,1,2,3

# FireSight on specific GPUs
bash scripts/train_ddp.sh configs/firesight_s_1024.yaml 0,1,2,3
```

### Evaluation

```bash
python eval.py --config configs/ccpe_single_1024.yaml --checkpoint runs/ccpe_single_1024/best.pth --gpu 0
python eval.py --config configs/firesight_s_1024.yaml --checkpoint runs/firesight_s_1024/best.pth --split test
```

## Key Innovations

### 1. Deformable Contrast Module (DCM)
Replaces CCPE's fixed pixel shifts with **learned deformable offsets**.
The network discovers optimal spatial contexts for smoke vs fire discrimination.
Advantages: content-adaptive, directional freedom, fewer parameters.

### 2. Frequency Attention Module (FAM)
Decomposes features into frequency bands and reweights channels.
Smoke has low-frequency body + mid-frequency edges; fire has high-frequency boundaries.

### 3. Transparency Module (TM)
Models semi-transparent smoke via multi-scale feature subtraction:
`enhanced = x + α · (x - smooth(x))`. Highlights what differs from local context.

### 4. Temporal Motion Fusion
Cross-frame attention guided by motion map. Focuses on regions where temporal
change indicates smoke growth. Works with 60s and 360s capture intervals.

## Project Structure

```
SAFE-Det/
├── train.py                 # DDP training (torchrun compatible)
├── eval.py                  # COCO mAP evaluation
├── configs/                 # YAML configs for all variants
├── models/
│   ├── ccpe_module.py       # Cross Contrast Patch Embedding
│   ├── swin_ccpe.py         # Swin Transformer + CCPE backbone
│   ├── neck.py              # YOLOX PAFPN
│   ├── head.py              # YOLOX head + SNSM
│   ├── detector.py          # CCPE detector (baseline)
│   └── firesight/           # Novel FireSight modules
│       ├── deformable_contrast.py
│       ├── frequency_attention.py
│       ├── transparency.py
│       ├── temporal_fusion.py
│       └── firesight_detector.py
├── utils/
│   ├── dataset.py           # COCO/YOLO dataset + augmentation
│   └── assigner.py          # SimOTA assignment
└── scripts/
    └── train_ddp.sh         # Multi-GPU launch script
```

## References

```bibtex
@article{wang2025wildfire,
  title={Wildfire Smoke Detection System: Model Architecture, Training Mechanism, and Dataset},
  author={Wang, Chong and Xu, Chen and Akram, Adeel and Wang, Zhong and Shan, Zhilin and Zhang, Qixing},
  journal={International Journal of Intelligent Systems},
  volume={2025},
  year={2025}
}
```
