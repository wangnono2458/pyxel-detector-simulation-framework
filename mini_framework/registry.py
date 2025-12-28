from __future__ import annotations

from typing import Callable, Dict

_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    def decorator(func: Callable):
        _REGISTRY[name] = func
        return func

    return decorator


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Model '{name}' is not registered")
    return _REGISTRY[name]


def list_models() -> Dict[str, str]:
    return {name: f"{func.__module__}:{func.__name__}" for name, func in _REGISTRY.items()}
