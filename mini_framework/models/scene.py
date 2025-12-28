from __future__ import annotations

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


@register("generate_stub_scene")
def generate_stub_scene(detector: Detector, height: int = 16, width: int = 16, level: float = 10.0):
    """Create a simple gradient scene array."""
    y = np.arange(height).reshape(-1, 1)
    x = np.arange(width).reshape(1, -1)
    detector.scene.data = level + y + x
    detector.scene.metadata["generated"] = True


@register("ir_background")
def ir_background(detector: Detector, sky_level: float = 50.0):
    if detector.scene.data is None:
        detector.scene.data = np.full((32, 32), sky_level, dtype=float)
    else:
        detector.scene.data = detector.scene.data + sky_level
    detector.scene.metadata["sky_level"] = sky_level
