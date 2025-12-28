from __future__ import annotations

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


@register("read_noise_and_gain")
def read_noise_and_gain(
    detector: Detector,
    read_noise: float = 3.0,
    system_gain: float = 1.0,
    bias: float = 100.0,
    seed: int | None = None,
):
    if detector.charge.electrons is None:
        raise ValueError("Charge array missing; run charge generation")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, read_noise, size=detector.charge.electrons.shape)
    volts = (detector.charge.electrons + noise) / system_gain + bias
    detector.pixel.signal = volts
    detector.pixel.metadata.update({"read_noise": read_noise, "gain": system_gain, "bias": bias})
