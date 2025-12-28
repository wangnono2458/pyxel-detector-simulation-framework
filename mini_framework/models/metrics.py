from __future__ import annotations

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


@register("compute_metrics")
def compute_metrics(detector: Detector):
    if detector.image.array is None:
        raise ValueError("Image array missing; run readout first")
    arr = detector.image.array
    detector.image.metadata["metrics"] = {
        "mean": float(np.nanmean(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }
