from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from mini_framework import buckets


def create_run_dir(root: Path) -> Path:
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_config(run_dir: Path, config: dict) -> None:
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def dump_manifest(run_dir: Path, manifest: Dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def dump_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def dump_snapshot(run_dir: Path, stage: str, detector) -> None:
    stage_dir = run_dir / "snapshots" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "scene": buckets.summarize_bucket(detector.scene),
        "photon": buckets.summarize_bucket(detector.photon),
        "charge": buckets.summarize_bucket(detector.charge),
        "image": buckets.summarize_bucket(detector.image),
    }
    (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
