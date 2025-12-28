from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import List, Optional

from mini_framework.stage import Stage


@dataclass
class DetectionPipeline:
    """Fixed-order detection pipeline mirroring Pyxel-like stages."""

    stages: List[Stage] = field(default_factory=list)

    ORDER: tuple[str, ...] = (
        "scene_generation",
        "photon_collection",
        "phasing",
        "charge_generation",
        "charge_collection",
        "charge_transfer",
        "charge_measurement",
        "signal_transfer",
        "readout_electronics",
        "data_processing",
    )

    def __iter__(self) -> Iterator[Stage]:
        return iter(self.stages)

    @classmethod
    def from_stage_map(cls, stage_map: dict[str, Stage]) -> "DetectionPipeline":
        ordered: List[Stage] = []
        for name in cls.ORDER:
            if name in stage_map:
                ordered.append(stage_map[name])
        return cls(stages=ordered)
