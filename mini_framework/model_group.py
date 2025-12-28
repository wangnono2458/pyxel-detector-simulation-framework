from __future__ import annotations

import logging
from typing import Iterator, Optional, Sequence

from mini_framework.model_function import ModelFunction


class ModelGroup:
    """Collection of ModelFunction executed in order if enabled."""

    def __init__(self, models: Sequence[ModelFunction], name: str):
        self._log = logging.getLogger(__name__)
        self.models = list(models)
        self.name = name

    def __iter__(self) -> Iterator[ModelFunction]:
        for model in self.models:
            if model.enabled:
                yield model

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def run(self, detector, debug: bool = False):
        for model in self:
            self._log.info("Running model '%s' in group '%s'", model.name, self.name)
            model(detector)
