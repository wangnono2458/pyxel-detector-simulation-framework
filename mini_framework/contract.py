from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from mini_framework import buckets


@dataclass
class Contract:
    requires: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)

    def validate_inputs(self, detector) -> None:
        for name in self.requires:
            obj = getattr(detector, name, None)
            if obj is None:
                raise ValueError(f"Contract requires bucket '{name}' but detector has None")
            primary = buckets._get_primary_array(obj)
            if primary is None:
                raise ValueError(f"Contract requires bucket '{name}' to be populated before stage")

    def validate_outputs(self, detector) -> None:
        for name in self.produces:
            obj = getattr(detector, name, None)
            if obj is None:
                raise ValueError(f"Contract expects bucket '{name}' to exist after stage")
            primary = buckets._get_primary_array(obj)
            if primary is None:
                raise ValueError(f"Contract expects bucket '{name}' to be populated after stage")
