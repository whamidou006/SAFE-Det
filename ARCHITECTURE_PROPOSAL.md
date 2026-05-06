# FireSight: A Novel Architecture for Fire/Smoke Detection

## Motivation

Current fire/smoke detection models either:
1. Use generic architectures (YOLO, DETR) with no domain adaptation
2. Add smoke-specific modules but use outdated backbones (CCPE uses Swin-T from 2021)

We propose **FireSight** — combining the best domain-specific innovations from CCPE
with modern (2024-2025) architecture advances for a purpose-built fire/smoke detector.

---

## Literature Review: Key Ideas to Combine

### From CCPE (Wang et al., IJIS 2025)
- **Cross Contrast Patch Embedding**: Multi-scale spatial gradients (1-128px shifts)
  capture soft smoke edges that standard convolutions miss
- **SNSM**: Separable negative sampling addresses bg/fg confusion
- **Limitation**: Swin-T backbone is small/old, no modern self-supervised pretraining

### From RF-DETR (Zhao et al., 2024)
- **DINOv2 backbone**: Self-supervised ViT features are far stronger than supervised
- **Lightweight decoder**: Efficient DETR head with deformable attention
- **Limitation**: Generic — no smoke-specific feature enhancement

### From D-FINE (Peng et al., 2024)
- **Fine-grained distribution refinement**: Better box regression via distribution
- **HGNetv2**: Efficient CNN backbone with large receptive field
- **Limitation**: No temporal modeling, no smoke-specific modules

### From EfficientViT (Cai et al., 2024)
- **Linear attention**: O(n) instead of O(n²) for high-resolution features
- **Multi-scale token interaction**: Efficient cross-resolution fusion
- **Use case**: When real-time inference (>30fps) is needed

### From InternImage / DCNv4 (Wang et al., 2024)
- **Deformable convolutions v4**: Adaptive receptive field, faster than attention
- **Key insight**: Deformable kernels can learn to follow smoke plume shapes

### From Temporal Models (VideoMAE, TimeSformer)
- **Temporal attention**: Cross-frame feature correlation
- **Key for smoke**: Smoke is inherently temporal — it grows, drifts, dissipates
- **Our advantage**: ALERTCalifornia has 60s intervals = natural temporal signal

### From Frequency-Domain Detection (FcaNet, FSDR)
- **Channel attention in frequency domain**: Smoke has distinct frequency signature
- **Low-frequency = diffuse body, high-frequency = edges**
- **Separation helps**: Process smoke body and edges differently

---

## Proposed Architecture: FireSight

### Design Philosophy
1. **Strong pretrained backbone** — leverage DINOv2/InternImage self-supervised features
2. **Smoke-aware feature enhancement** — adapt CCPE contrast idea to work at feature level
3. **Temporal aggregation** — lightweight cross-frame attention for sequential cameras
4. **Efficient detection head** — DETR-style (no NMS) with smoke/fire specialization

### Architecture Diagram

```
Input (1024×1024×3) ──or── Multi-frame (1024×1024×6)
        │
        ▼
┌─────────────────────────────────────────────┐
│  BACKBONE: DINOv2-Small (ViT-S/14)         │
│  - Frozen or fine-tuned                      │
│  - Multi-scale via FPN adapter              │
│  - Output: {P3, P4, P5} at strides 8,16,32 │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  SMOKE-AWARE FEATURE ENHANCEMENT (SAFE)    │
│                                             │
│  1. Deformable Contrast Module (DCM):      │
│     - Learned offsets instead of fixed      │
│       shifts (1,2,4,...128)                  │
│     - Directional: separate H/V/diagonal    │
│     - Applied at P3 (highest res) only      │
│                                             │
│  2. Frequency Attention Module (FAM):      │
│     - DCT decomposition of features        │
│     - Separate paths for low/high freq     │
│     - Low-freq → smoke body detection      │
│     - High-freq → edge/boundary detection  │
│     - Channel attention reweighting        │
│                                             │
│  3. Transparency Modeling (TM):            │
│     - Feature subtraction: F - blur(F)     │
│     - Captures what's "different" from bg  │
│     - Critical for semi-transparent smoke  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  TEMPORAL FUSION (optional, for sequences) │
│                                             │
│  - Cross-attention between current and     │
│    previous frame features                  │
│  - Motion-guided: attend to regions with   │
│    temporal change (smoke growth areas)     │
│  - Lightweight: only 2 transformer layers  │
│  - Skip connection: works without temporal │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  DETECTION HEAD: Hybrid DETR               │
│                                             │
│  Option A: Lightweight DETR decoder        │
│  - 100 learned queries                     │
│  - 3 decoder layers with deformable attn   │
│  - No NMS needed                           │
│                                             │
│  Option B: YOLOX head + SNSM              │
│  - Faster inference                        │
│  - SNSM for hard-negative handling         │
│  - SimOTA assignment                       │
│                                             │
│  Output: boxes + class (smoke/fire) + conf │
└─────────────────────────────────────────────┘
```

---

## Key Novel Contributions

### 1. Deformable Contrast Module (DCM) — Evolution of CCPE

CCPE uses **fixed shifts** (1,2,4,...128 pixels) in H and V directions.
Our DCM uses **learned deformable offsets**:

```python
class DeformableContrast(nn.Module):
    """
    Instead of fixed pixel shifts, learn where to sample contrast.
    The network learns the optimal spatial context for smoke vs fire.
    """
    def __init__(self, channels, num_points=8):
        self.offset_conv = nn.Conv2d(channels, 2 * num_points, 3, 1, 1)
        self.contrast_conv = nn.Conv2d(channels * num_points, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        offsets = self.offset_conv(x)  # (B, 2*K, H, W)
        # Sample at learned offsets, compute contrast
        contrasts = deformable_sample_and_subtract(x, offsets)
        return self.norm(self.contrast_conv(contrasts)) + x
```

**Advantages over CCPE:**
- Adapts to content (smoke plume shape vs fire flickering)
- Fewer params (no 8×2 separate conv branches)
- Can learn diagonal/curved patterns (not just H/V)

### 2. Frequency Attention Module (FAM)

Smoke has a unique frequency signature:
- **Low-freq**: Diffuse opacity (body)
- **Mid-freq**: Gradual gradients (edges)
- **High-freq**: Sharp boundaries (fire, not smoke)

```python
class FrequencyAttention(nn.Module):
    def __init__(self, channels, num_freq=8):
        # DCT basis functions
        self.freq_weights = nn.Parameter(torch.ones(num_freq))
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Decompose into frequency bands
        freq_bands = dct_decompose(x, self.freq_weights)
        # Reweight channels based on frequency content
        attn = self.channel_attn(freq_bands)
        return x * attn
```

### 3. Temporal Motion-Guided Attention

For cameras with sequential captures (60s or 360s intervals):

```python
class TemporalMotionAttention(nn.Module):
    """
    Attend to regions where temporal change indicates smoke growth.
    Motion map guides spatial attention.
    """
    def __init__(self, channels):
        self.motion_conv = nn.Conv2d(channels * 2, 1, 3, 1, 1)
        self.cross_attn = nn.MultiheadAttention(channels, 4)

    def forward(self, current_feat, prev_feat):
        # Compute motion/change map
        motion = self.motion_conv(torch.cat([current_feat, prev_feat], dim=1))
        motion_mask = motion.sigmoid()  # High where change occurs

        # Cross-attention guided by motion
        q = current_feat * motion_mask  # Focus on changing regions
        k = v = prev_feat
        fused = self.cross_attn(q, k, v)
        return current_feat + fused
```

---

## Model Variants

| Variant | Backbone | SAFE | Temporal | Head | Params | Target |
|---------|----------|------|----------|------|--------|--------|
| FireSight-S | DINOv2-S/14 | DCM+FAM | No | DETR-3L | ~35M | Accuracy |
| FireSight-T | EfficientViT-L | DCM | No | YOLOX+SNSM | ~25M | Speed |
| FireSight-ST | DINOv2-S/14 | DCM+FAM+TM | Yes | DETR-3L | ~40M | Temporal |
| FireSight-L | DINOv2-B/14 | Full | Yes | DETR-6L | ~80M | Best |

---

## Comparison with Existing Approaches

| Feature | YOLO11 | D-FINE | RF-DETR | CCPE | **FireSight** |
|---------|--------|--------|---------|------|---------------|
| Backbone pretrain | ImageNet-sup | ImageNet-sup | DINOv2-SSL | ImageNet-sup | **DINOv2-SSL** |
| Smoke-specific | ✗ | ✗ | ✗ | CCPE | **DCM+FAM+TM** |
| Temporal | ✗ | ✗ | ✗ | 6ch concat | **Motion attn** |
| Hard-neg handling | ✗ | ✗ | ✗ | SNSM | **SNSM+** |
| Deformable features | ✗ | Def. Attn | Def. Attn | ✗ | **DCM** |
| Frequency domain | ✗ | ✗ | ✗ | ✗ | **FAM** |
| Detection head | Anchor-free | DETR | DETR | YOLOX | **Hybrid** |
| Resolution | 1024 | 640-1024 | 560-1024 | 1024 | **1024** |

---

## Implementation Roadmap

### Phase 1: Baseline with Modern Backbone
- Replace Swin-T with DINOv2-S in CCPE framework
- Keep CCPE module and SNSM
- Compare against original CCPE → validate backbone improvement

### Phase 2: Deformable Contrast Module
- Replace fixed-shift CCPE with learned deformable contrast
- Ablation: DCM vs CCPE at same compute budget
- Expected: better mAP with fewer params

### Phase 3: Frequency Attention
- Add FAM after DCM
- Ablation: with/without FAM
- Expected: better smoke/cloud discrimination

### Phase 4: Temporal Fusion
- Add motion-guided cross-frame attention
- Train on sequences from our temporal dataset
- Expected: significant improvement for early smoke detection

### Phase 5: Detection Head Optimization
- Compare YOLOX+SNSM vs lightweight DETR decoder
- DETR should be better for overlapping smoke/fire
- Optimize for inference speed

---

## Expected Improvements

Based on component analysis:

| Component | Expected mAP Gain | Evidence |
|-----------|-------------------|----------|
| DINOv2 → Swin-T | +5-8% | SSL features consistently outperform |
| DCM → fixed CCPE | +2-3% | Adaptive > fixed spatial sampling |
| FAM | +1-2% | Frequency discrimination for smoke |
| Temporal | +3-5% | Smoke is inherently temporal |
| SNSM+ | +1-2% | Better hard-negative handling |
| **Total expected** | **+12-20%** | vs baseline CCPE |

---

## Research Questions

1. **Does CCPE's spatial contrast help when combined with DINOv2?**
   - DINOv2 already has strong feature representations
   - Hypothesis: Yes, because DINOv2 is generic while DCM is smoke-specific

2. **Fixed vs learned contrast: which is better for smoke?**
   - CCPE argues fixed multi-scale is sufficient
   - We argue deformable is better (adapts to plume shapes)

3. **Is frequency decomposition useful given modern backbones?**
   - Smoke's frequency signature may already be captured by deep features
   - Ablation study needed

4. **How much does temporal information help for early detection?**
   - Our dataset has 60s and 360s intervals
   - Early smoke is tiny → temporal growth is the strongest signal

5. **DETR vs anchor-free for smoke detection?**
   - Smoke overlaps with itself and fire
   - DETR handles duplicates better (no NMS)
   - But YOLOX with SNSM handles hard negatives better

---

## References

1. Wang et al., "Wildfire Smoke Detection System: Model Architecture, Training Mechanism, and Dataset", IJIS 2025
2. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024
3. Zhao et al., "RF-DETR: Real-Time Detection Transformer", 2024
4. Peng et al., "D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement", 2024
5. Cai et al., "EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction", ICLR 2024
6. Wang et al., "InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions", CVPR 2023
7. Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021
8. Qin et al., "FcaNet: Frequency Channel Attention Networks", ICCV 2021
