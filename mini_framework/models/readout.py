from __future__ import annotations

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


@register("simple_adc_readout")
def simple_adc_readout(detector: Detector, gain: float = 1.0, offset: float = 0.0):
    if detector.charge.electrons is None:
        raise ValueError("Charge data is missing; run charge stage first")
    signal = detector.charge.electrons * gain + offset
    detector.image.array = signal.astype(np.float32)
    detector.image.metadata.update({"gain": gain, "offset": offset})


@register("add_read_noise")
def add_read_noise(detector: Detector, sigma: float = 1.0, seed: int | None = None):
    if detector.image.array is None:
        raise ValueError("Image array missing; run readout first")
    rng = np.random.default_rng(seed)
    detector.image.array = detector.image.array + rng.normal(0, sigma, size=detector.image.array.shape)
    detector.image.metadata["read_noise"] = sigma


@register("apply_nonlinearity")
def apply_nonlinearity(detector: Detector, saturation: float = 65535.0):
    if detector.image.array is None:
        raise ValueError("Image array missing; run readout first")
    detector.image.array = np.clip(detector.image.array, 0, saturation)
    detector.image.metadata["saturation"] = saturation
