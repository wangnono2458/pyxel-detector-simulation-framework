from __future__ import annotations

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


@register("qe_shotnoise")
def qe_shotnoise(detector: Detector, quantum_efficiency: float = 0.9, seed: int | None = None):
    if detector.photon.photons is None:
        raise ValueError("Photon data missing; run photon_collection first")
    rng = np.random.default_rng(seed)
    expected_e = detector.photon.photons * quantum_efficiency
    detector.charge.electrons = rng.poisson(expected_e)
    detector.charge.metadata.update({"qe": quantum_efficiency})


@register("add_dark_current")
def add_dark_current(detector: Detector, dark_current: float = 0.1):
    if detector.charge.electrons is None:
        raise ValueError("Charge array missing; run charge_generation first")
    detector.charge.electrons = detector.charge.electrons + dark_current * detector.integration_time
    detector.charge.metadata["dark_current"] = dark_current
