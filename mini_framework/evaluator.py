from __future__ import annotations

import importlib
from typing import Callable


class ReferenceError(Exception):
    """Raised when a reference string cannot be resolved."""


REF_PREFIX = "${"
REF_SUFFIX = "}"


def evaluate_reference(reference_str: str) -> Callable:
    """Resolve "package.module:function" into a callable."""
    if not reference_str or ":" not in reference_str:
        raise ReferenceError("Function reference must be 'module:function'.")

    module_path, func_name = reference_str.split(":", maxsplit=1)
    module = importlib.import_module(module_path)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ReferenceError(f"{reference_str!r} is not a callable reference")
    return func


def is_reference(value: str) -> bool:
    return value.startswith(REF_PREFIX) and value.endswith(REF_SUFFIX)


def strip_reference(value: str) -> str:
    return value[len(REF_PREFIX) : -len(REF_SUFFIX)]
