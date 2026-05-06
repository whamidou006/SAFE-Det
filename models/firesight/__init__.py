"""
FireSight Novel Modules for Fire/Smoke Detection.

Contains:
- DeformableContrast: Learned spatial contrast (evolution of CCPE)
- FrequencyAttention: Frequency-domain channel attention
- TransparencyModeling: Feature subtraction for semi-transparent objects
- TemporalMotionAttention: Cross-frame motion-guided attention
"""

from .deformable_contrast import DeformableContrastModule
from .frequency_attention import FrequencyAttentionModule
from .transparency import TransparencyModule
from .temporal_fusion import TemporalMotionFusion
from .firesight_detector import FireSightDetector
