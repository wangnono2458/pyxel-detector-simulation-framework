from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mini_framework import buckets


@dataclass
class Detector:
    """Simple detector container holding simulation buckets."""

    integration_time: float = 1.0
    time: float = 0.0
    scene: buckets.Scene = field(default_factory=buckets.Scene)
    photon: buckets.Photon = field(default_factory=buckets.Photon)
    charge: buckets.Charge = field(default_factory=buckets.Charge)
    image: buckets.Image = field(default_factory=buckets.Image)

    def reset(self) -> None:
        self.scene = buckets.Scene()
        self.photon = buckets.Photon()
        self.charge = buckets.Charge()
        self.image = buckets.Image()

    def bucket_iter(self):
        yield self.scene
        yield self.photon
        yield self.charge
        yield self.image
