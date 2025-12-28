from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, List

from mini_framework.contracts import Contract
from mini_framework.model_function import ModelFunction


@dataclass
class Stage:
    name: str
    models: List[ModelFunction] = field(default_factory=list)
    contract: Contract = field(default_factory=Contract)

    def __iter__(self) -> Iterator[ModelFunction]:
        for model in self.models:
            if model.enabled:
                yield model

    def run(self, detector, *, debug: bool = False):
        log = logging.getLogger(__name__)
        log.info("Stage '%s' starting", self.name)
        self.contract.validate_inputs(detector)
        for model in self:
            log.info("  Model '%s'", model.name)
            model(detector)
        self.contract.validate_outputs(detector)
        if debug:
            log.info("Stage '%s' completed", self.name)
