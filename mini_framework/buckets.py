from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Scene:
    data: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Photon:
    photons: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Charge:
    electrons: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pixel:
    signal: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Image:
    array: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


BucketType = Scene | Photon | Charge | Pixel | Image


def summarize_bucket(bucket: BucketType) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": bucket.__class__.__name__}
    array = _get_primary_array(bucket)

    if array is None:
        summary.update({"empty": True})
        return summary

    summary.update(
        {
            "empty": False,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(np.nanmin(array)),
            "max": float(np.nanmax(array)),
            "mean": float(np.nanmean(array)),
        }
    )
    return summary


def _get_primary_array(bucket: BucketType) -> Optional[np.ndarray]:
    if isinstance(bucket, Scene):
        return bucket.data
    if isinstance(bucket, Photon):
        return bucket.photons
    if isinstance(bucket, Charge):
        return bucket.electrons
    if isinstance(bucket, Pixel):
        return bucket.signal
    if isinstance(bucket, Image):
        return bucket.array
    return None
