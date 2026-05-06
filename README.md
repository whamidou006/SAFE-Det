# SAFE-Det — Smoke-Aware Feature Enhancement Detector

Novel fire/smoke detection architecture for wildfire early-warning systems.
Combines ideas from CCPE (Wang et al., IJIS 2025) with modern vision transformers
and domain-specific innovations (DCM/FAM/TM/Temporal), and supports a D-FINE
DETR-style head for production parity with the user's Condor-evaluation stack.

---

## 1. Architectures

### CCPE Baseline (re-implemented)
- **Backbone**: Swin Transformer + Cross Contrast Patch Embedding (CCPE)
- **Neck**: YOLOX-style PAFPN
- **Head**: YOLOX + Separable Negative Sampling Mechanism (SNSM)

### FireSight (novel)
- **Backbone**: DINOv2 ViT (self-supervised, much stronger features)
- **SAFE modules** (any combination): DCM (Deformable Contrast), FAM (Frequency
  Attention), TM (Transparency), Temporal Motion Fusion
- **Head**: YOLOX (default) **or** D-FINE decoder (`head_type: dfine`)

---

## 2. Environment

```bash
# Conda env with torch ≥ 2.0, torchvision ≥ 0.20, opencv, scipy, pyyaml
conda activate /home/whamidouche/ssdprivate/conda_envs/condor-bench

# Repo deps
cd /home/whamidouche/ssdprivate/SAFE-Det
pip install -r requirements.txt

# Run the test suite (67 tests; should pass before training)
pytest -q
```

> **D-FINE head only.** The `head_type: dfine` option imports the user's
> `engine.rtv4` package from a local Condor-evaluation checkout. Default
> search path is `/home/whamidouche/ssdprivate/Condor-evaluation`. Override
> with `cfg.model.dfine_source: /alt/path` or `export
> CONDOR_EVALUATION_ROOT=/alt/path`.

---

## 3. Dataset

Default ALERT-California layout (YOLO format), pre-pointed in every config:

```
/home/whamidouche/ssdprivate/datasets/data_v3_and_imerit_upto_april06_2classes_40_unseen_cams_cleaned/
├── train/{images,labels}        # 159,211 frames
├── test_random/{images,labels}  #   8,801 frames (same cameras as train)
└── test_disjoint/{images,labels}#   4,967 frames (40 unseen cameras)
```

Classes: `smoke=0, fire=1`. To use a different dataset, edit the four
`*_img_dir` / `*_label_dir` lines at the top of any config.

---

## 4. Available configs

| Config | Backbone | Modules | Head | Loss / assigner | Params |
|---|---|---|---|---|---|
| `ccpe_single_1024.yaml`        | Swin-T + CCPE | — | YOLOX | CIoU + SimOTA | ~30M |
| `ccpe_multi_1024.yaml`         | Swin-T + CCPE | 6-ch concat | YOLOX | CIoU + SimOTA | ~30M |
| `ccpe_base_1024.yaml`          | Swin-B + CCPE | — | YOLOX | CIoU + SimOTA | ~88M |
| `firesight_s_1024.yaml`        | DINOv2-S | DCM+FAM+TM | YOLOX | CIoU + SimOTA | ~29M |
| `firesight_st_1024.yaml`       | DINOv2-S | DCM+FAM+TM + Temporal | YOLOX | CIoU + SimOTA | ~31M |
| `firesight_s_nwd_tal_1024.yaml`| DINOv2-S | DCM+FAM+TM | YOLOX | **NWD+CIoU + TAL** | ~29M |
| `firesight_dfine_1024.yaml`    | DINOv2-S | DCM+FAM+TM | **D-FINE** | Hungarian + Focal/L1/GIoU | ~35M |

---

## 5. Training — one command per variant

All configs use bf16 autocast under the hood. Single-GPU is for smoke-tests;
production runs should use `train_ddp.sh` for multi-GPU DDP.

### 5.1 CCPE baselines (Swin + CCPE + YOLOX)

```bash
# Single-GPU
python train.py --config configs/ccpe_single_1024.yaml --gpu 0

# 4-GPU DDP
bash scripts/train_ddp.sh configs/ccpe_single_1024.yaml 0,1,2,3

# Multi-frame variant (concatenated 6-channel input)
bash scripts/train_ddp.sh configs/ccpe_multi_1024.yaml 0,1,2,3

# Larger backbone (Swin-B)
bash scripts/train_ddp.sh configs/ccpe_base_1024.yaml 0,1,2,3
```

### 5.2 FireSight — DINOv2 + SAFE modules, YOLOX head

```bash
# Single-frame
bash scripts/train_ddp.sh configs/firesight_s_1024.yaml 0,1,2,3

# Single-frame + temporal motion fusion (needs sequential frames)
bash scripts/train_ddp.sh configs/firesight_st_1024.yaml 0,1,2,3
```

### 5.3 FireSight — NWD loss + Task-Aligned assigner ablation

Drop-in replacement for `firesight_s_1024` that swaps the bbox loss to
**NWD+CIoU** (paper: *A Normalized Gaussian Wasserstein Distance for Tiny
Object Detection*, CVPR 2022) and the label assigner to **TAL** (TOOD, ICCV
2021). Useful for the small-smoke recall regime described in
`safedet_testable_proposals.md`.

```bash
bash scripts/train_ddp.sh configs/firesight_s_nwd_tal_1024.yaml 0,1,2,3
```

Tunable in the config's `loss:` block:
- `bbox_loss_type`: `ciou` (default) | `nwd` | `mixed`
- `nwd_constant`: gaussian kernel size (default `12.8`)
- `nwd_mix_weight`: NWD weight when `mixed` (default `0.5`)
- `assigner`: `simota` (default) | `tal` | `dsla`

### 5.4 FireSight — D-FINE decoder head

This path uses the rtv4 `DFINETransformer` decoder with the Hungarian-matched
DETR criterion (`RTv4Criterion`) and the rtv4 `PostProcessor` for evaluation.
Both training and eval loops dispatch automatically on `head_type: dfine`.

Pre-flight (only once):

```bash
# Make sure Condor-evaluation is reachable from SAFE-Det
export CONDOR_EVALUATION_ROOT=/home/whamidouche/ssdprivate/Condor-evaluation
ls $CONDOR_EVALUATION_ROOT/engine/rtv4/dfine_decoder.py    # must exist
```

Train:

```bash
# 4-GPU DDP, recommended
bash scripts/train_ddp.sh configs/firesight_dfine_1024.yaml 0,1,2,3

# Single-GPU smoke-test
python train.py --config configs/firesight_dfine_1024.yaml --gpu 0
```

Customise via the config:

```yaml
model:
  head_type: dfine
  dfine_source: /alt/Condor-evaluation       # optional override
  dfine_kwargs: { num_queries: 300, num_decoder_layers: 6 }
loss:
  dfine_weight_dict: { loss_focal: 1.0, loss_bbox: 5.0, loss_giou: 2.0 }
  dfine_losses: [focal, boxes]               # add 'vfl' / 'mal' / 'distill'
  dfine_reg_max: 32
```

---

## 6. Evaluation

Same `eval.py` for every architecture — it auto-dispatches on `head_type`
(YOLOX → greedy NMS, D-FINE → rtv4 `PostProcessor`).

```bash
# Random-split (same cameras as train)
python eval.py --config configs/firesight_s_1024.yaml \
               --checkpoint runs/firesight_s_1024/best.pth \
               --split val --gpu 0

# Disjoint-camera generalisation test
python eval.py --config configs/firesight_dfine_1024.yaml \
               --checkpoint runs/firesight_dfine_1024/best.pth \
               --split test --gpu 0
```

Outputs per-class AP50 + precision/recall + mAP50.

---

## 7. Resuming a run

```bash
python train.py --config configs/firesight_s_1024.yaml \
                --resume runs/firesight_s_1024/epoch_29.pth --gpu 0
```

Optimizer state and `best_loss` are restored.

---

## 8. Project structure

```
SAFE-Det/
├── train.py                 # DDP training (auto-dispatches YOLOX vs D-FINE)
├── eval.py                  # COCO mAP eval (auto-dispatches NMS vs PostProcessor)
├── configs/                 # YAML configs for all variants
├── models/
│   ├── ccpe_module.py       # Cross Contrast Patch Embedding
│   ├── swin_ccpe.py         # Swin Transformer + CCPE backbone
│   ├── neck.py              # YOLOX PAFPN
│   ├── head.py              # YOLOX head + SNSM
│   ├── detector.py          # CCPE detector (baseline)
│   ├── losses_nwd.py        # Normalized Wasserstein Distance + bbox dispatcher
│   └── firesight/
│       ├── firesight_detector.py
│       ├── deformable_contrast.py
│       ├── frequency_attention.py
│       ├── transparency.py
│       ├── temporal_fusion.py
│       ├── dfine_head.py     # DFINEHeadAdapter (lazy rtv4 import)
│       └── dfine_runtime.py  # Hungarian criterion + PostProcessor wrappers
├── utils/
│   ├── dataset.py            # YOLO-format dataset + augmentation
│   └── assigner.py           # SimOTA + TAL + DSLA + build_assigner()
├── tests/                    # 67 pytest tests — run before training
└── scripts/
    └── train_ddp.sh          # Multi-GPU launch script
```

---

## 9. Quick checklist before launching training

1. Activate the conda env: `conda activate .../condor-bench`
2. `cd /home/whamidouche/ssdprivate/SAFE-Det`
3. `pytest -q` — must show **67 passed**
4. (D-FINE only) `ls $CONDOR_EVALUATION_ROOT/engine/rtv4/dfine_decoder.py`
5. Pick a config from §4 and launch with the matching command from §5
6. Outputs land in `runs/<config-name>/`

---

## References

```bibtex
@article{wang2025wildfire,
  title={Wildfire Smoke Detection System: Model Architecture, Training Mechanism, and Dataset},
  author={Wang, Chong and Xu, Chen and Akram, Adeel and Wang, Zhong and Shan, Zhilin and Zhang, Qixing},
  journal={International Journal of Intelligent Systems},
  volume={2025},
  year={2025}
}
@inproceedings{xu2022nwd,
  title={A Normalized Gaussian Wasserstein Distance for Tiny Object Detection},
  author={Xu, Chang and Wang, Jinwang and Yang, Wen and Yu, Lei},
  booktitle={CVPR Workshops},
  year={2022}
}
@inproceedings{feng2021tood,
  title={TOOD: Task-aligned One-stage Object Detection},
  author={Feng, Chengjian and Zhong, Yujie and Gao, Yu and Scott, Matthew R and Huang, Weilin},
  booktitle={ICCV},
  year={2021}
}
@inproceedings{peng2024dfine,
  title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
  author={Peng, Yansong and Li, Hebei and others},
  booktitle={ICLR},
  year={2025}
}
```
