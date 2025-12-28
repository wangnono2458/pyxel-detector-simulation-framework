from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

ENV_PATTERN = re.compile(r"\$\{env:([^}:]+)(?::([^}]*))?\}")
REF_PATTERN = re.compile(r"\$\{([^}]+)\}")


def merge_dicts(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_dicts(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = deepcopy(value)
    return result


def resolve_env(value: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        var, default = match.group(1), match.group(2)
        return os.environ.get(var, default or "")

    return ENV_PATTERN.sub(_replace, value)


def resolve_refs_in_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        if ENV_PATTERN.search(obj):
            obj = resolve_env(obj)
        if REF_PATTERN.search(obj):
            # Placeholder remaining for later resolution using context
            return obj
        return obj
    if isinstance(obj, list):
        return [resolve_refs_in_obj(v) for v in obj]
    if isinstance(obj, dict):
        return {k: resolve_refs_in_obj(v) for k, v in obj.items()}
    return obj


def resolve_placeholders(obj: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(obj, str):
        def _replace(match: re.Match[str]) -> str:
            path = match.group(1)
            try:
                value = _lookup(context, path)
                return str(value)
            except Exception:
                return match.group(0)
        return REF_PATTERN.sub(_replace, obj)
    if isinstance(obj, list):
        return [resolve_placeholders(v, context) for v in obj]
    if isinstance(obj, dict):
        return {k: resolve_placeholders(v, context) for k, v in obj.items()}
    return obj


def _lookup(context: Mapping[str, Any], path: str) -> Any:
    parts = path.split(".")
    current: Any = context
    for part in parts:
        if isinstance(current, Mapping):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def load_yaml_with_inheritance(path: Path) -> dict:
    import yaml

    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    extends = data.pop("extends", None)
    if extends:
        base_path = (path.parent / extends).resolve()
        base_data = load_yaml_with_inheritance(base_path)
        data = merge_dicts(base_data, data)
    return data
