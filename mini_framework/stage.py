from __future__ import annotations

import logging
from typing import Iterator, Sequence

from mini_framework.contract import Contract
from mini_framework.model_function import ModelFunction


class Stage:
    def __init__(self, name: str, models: Sequence[ModelFunction], contract: Contract | None = None):
        self.name = name
        self.models = list(models)
        self.contract = contract or Contract()
        self._log = logging.getLogger(__name__)

    def __iter__(self) -> Iterator[ModelFunction]:
        for model in self.models:
            if model.enabled:
                yield model

    def run(self, detector, *, debug: bool = False):
        self._log.info("Stage '%s' starting", self.name)
        self.contract.validate_inputs(detector)
        for model in self:
            self._log.info("  Model '%s'", model.name)
            model(detector)
        self.contract.validate_outputs(detector)
        if debug:
            self._log.info("Stage '%s' completed", self.name)
