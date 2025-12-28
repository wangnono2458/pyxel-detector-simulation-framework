from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from mini_framework import buckets


@dataclass
class Contract:
    requires: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)

    def validate_inputs(self, detector) -> None:
        for name in self.requires:
            bucket = getattr(detector, name, None)
            if bucket is None:
                raise ValueError(f"Missing required bucket '{name}' before stage")
            if buckets._get_primary_array(bucket) is None:
                raise ValueError(f"Bucket '{name}' must be populated before stage")

    def validate_outputs(self, detector) -> None:
        for name in self.produces:
            bucket = getattr(detector, name, None)
            if bucket is None:
                raise ValueError(f"Missing produced bucket '{name}' after stage")
            if buckets._get_primary_array(bucket) is None:
                raise ValueError(f"Bucket '{name}' must be populated after stage")
