from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict

from mini_framework.detector import Detector
from mini_framework.evaluator import evaluate_reference, is_reference, strip_reference
from mini_framework.registry import get as registry_get


class ModelFunction:
    """Wrapper around a callable described in configuration."""

    def __init__(
        self,
        name: str,
        func: str,
        arguments: Dict[str, Any] | None = None,
        enabled: bool = True,
    ):
        self.name = name
        self._func_ref = func
        self.enabled = enabled
        self.arguments = arguments or {}
        self._log = logging.getLogger(__name__)
        self._func: Callable | None = None

    @property
    def func(self) -> Callable:
        if self._func is None:
            if ":" in self._func_ref:
                self._func = evaluate_reference(self._func_ref)
            else:
                self._func = registry_get(self._func_ref)
        return self._func

    def __call__(self, detector: Detector) -> None:
        call_args = resolve_arguments(self.arguments, detector)
        self._log.debug("Calling %s with args %s", self._func_ref, call_args)
        self.func(detector, **call_args)


def resolve_arguments(arguments: Dict[str, Any], detector: Detector) -> Dict[str, Any]:
    def _resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        if isinstance(value, str) and is_reference(value):
            path = strip_reference(value)
            return resolve_path(detector, path)
        if isinstance(value, str):
            try:
                # attempt numeric coercion
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                return copy.deepcopy(value)
        return copy.deepcopy(value)

    return _resolve(arguments)


def resolve_path(detector: Detector, path: str) -> Any:
    """Resolve simple dot-separated path from detector or its buckets."""
    parts = path.split(".")
    current: Any = detector
    for idx, part in enumerate(parts):
        if idx == 0 and part == "detector":
            current = detector
            continue
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            raise AttributeError(f"Cannot resolve reference '{path}'")
    return current
