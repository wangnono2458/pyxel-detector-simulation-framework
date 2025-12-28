from __future__ import annotations

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


@register("generate_stub_charge")
def generate_stub_charge(detector: Detector, quantum_efficiency: float = 1.0):
    if detector.photon.photons is None:
        raise ValueError("Photon data is missing; run photon stage first")
    detector.charge.electrons = detector.photon.photons * quantum_efficiency
    detector.charge.metadata["qe"] = quantum_efficiency


@register("add_dark_current")
def add_dark_current(detector: Detector, dark_current: float = 0.1):
    if detector.charge.electrons is None:
        if detector.photon.photons is None:
            raise ValueError("Photon data is missing; run photon stage first")
        detector.charge.electrons = np.zeros_like(detector.photon.photons)
    detector.charge.electrons = detector.charge.electrons + dark_current
    detector.charge.metadata["dark_current"] = dark_current
