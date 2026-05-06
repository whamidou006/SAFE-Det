# Fire-Smoke CCPE — Cross Contrast Patch Embedding for Fire/Smoke Detection

Standalone PyTorch implementation of CCPE (Wang et al., IJIS 2025) adapted for
the ALERTCalifornia fire/smoke detection dataset.

## Architecture

- **Backbone**: Swin Transformer Tiny + Cross Contrast Patch Embedding (CCPE)
- **Neck**: YOLOX-style PAFPN (in=[192,384,768], out=128)
- **Head**: YOLOX head with Separable Negative Sampling Mechanism (SNSM)
- **Training**: SimOTA assignment, Mosaic+MixUp augmentation, FP16

## Model Variants

| Variant | Input | Description |
|---------|-------|-------------|
| ccpe_single_1024 | 1024×1024 RGB | Single-frame detection |
| ccpe_multi_1024 | 1024×1024 6ch | Multi-frame (current + previous) |
| ccpe_swin_base | 1024×1024 RGB | Swin-Base backbone (larger) |

## Dataset

Uses COCO-format annotations:
- Train: 178,394 images, 2 classes (smoke=0, fire=1)
- Val: 9,888 images (test_random)
- Test: 4,967 images (test_disjoint, 40 unseen cameras)

## Usage

```bash
# Single-GPU training
python train.py --config configs/ccpe_single_1024.yaml --gpu 0

# Multi-GPU DDP training (4 GPUs)
bash scripts/train_ddp.sh configs/ccpe_single_1024.yaml 4

# Evaluation
python eval.py --config configs/ccpe_single_1024.yaml --checkpoint runs/best.pth
```

## Key Innovations from CCPE Paper

1. **CCPE Module**: Multi-scale spatial contrast (shifts 1-128px) in H/V directions
   captures fine smoke edges that standard patch embedding misses.

2. **SNSM**: Separable negative sampling — different strategies for images with/without
   objects. Addresses supervision signal confusion in smoke detection.

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
