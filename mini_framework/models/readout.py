from __future__ import annotations

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


@register("quantize_clip")
def quantize_clip(detector: Detector, bit_depth: int = 16, full_well: float = 65535.0):
    if detector.pixel.signal is None:
        raise ValueError("Pixel signal missing; run charge_measurement first")
    max_dn = float(2**bit_depth - 1)
    arr = np.clip(detector.pixel.signal, 0, min(full_well, max_dn))
    detector.image.array = np.round(arr).astype(np.uint16)
    detector.image.metadata.update({"bit_depth": bit_depth, "full_well": full_well})
