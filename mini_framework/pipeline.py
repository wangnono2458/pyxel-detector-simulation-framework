from __future__ import annotations

from collections.abc import Iterator
from typing import List

from mini_framework.stage import Stage


class DetectionPipeline:
    MODEL_GROUPS: tuple[str, ...] = (
        "scene",
        "photon",
        "charge",
        "readout",
        "postproc",
    )

    def __init__(self, stages: List[Stage]):
        self.stages = stages

    def __iter__(self) -> Iterator[Stage]:
        return iter(self.stages)
