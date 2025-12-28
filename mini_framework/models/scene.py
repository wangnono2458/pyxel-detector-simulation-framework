from __future__ import annotations

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


@register("point_sources")
def point_sources(
    detector: Detector,
    sources: list[dict] | None = None,
    background: float = 0.0,
):
    """Create a scene array with simple point sources.

    sources: list of {"x": float, "y": float, "flux": float}
    """
    rows, cols = detector.rows, detector.cols
    scene = np.full((rows, cols), background, dtype=float)
    if sources:
        for src in sources:
            x = int(src.get("x", cols // 2))
            y = int(src.get("y", rows // 2))
            flux = float(src.get("flux", 1.0))
            if 0 <= x < cols and 0 <= y < rows:
                scene[y, x] += flux
    detector.scene.data = scene
    detector.scene.metadata.update({"background": background, "num_sources": len(sources or [])})
