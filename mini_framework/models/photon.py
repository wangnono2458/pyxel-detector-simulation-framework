from __future__ import annotations

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


@register("collect_stub_photons")
def collect_stub_photons(detector: Detector, scale: float = 1.0):
    if detector.scene.data is None:
        raise ValueError("Scene data is missing; run scene stage first")
    detector.photon.photons = np.maximum(detector.scene.data * scale, 0.0)
    detector.photon.metadata["scale"] = scale


@register("ir_photon_throughput")
def ir_photon_throughput(detector: Detector, throughput: float = 0.8):
    if detector.scene.data is None:
        raise ValueError("Scene data missing")
    detector.photon.photons = detector.scene.data * throughput
    detector.photon.metadata["throughput"] = throughput
