from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from mini_framework.config import Configuration, build_pipeline, config_to_dict
from mini_framework.detector import Detector
from mini_framework.outputs import create_run_dir, dump_config, dump_manifest, dump_metrics
from mini_framework.processor import Processor
from mini_framework.resolver import merge_dicts

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, raw = pair.split("=", 1)
        parts = key.split(".")
        target = overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})  # type: ignore[assignment]
        target[parts[-1]] = json.loads(raw) if _looks_like_json(raw) else raw
    return overrides


def _looks_like_json(value: str) -> bool:
    return value.startswith("[") or value.startswith("{") or value.isdigit() or value.replace(".", "", 1).isdigit()


def run_exposure(config: Configuration, run_dir: Path) -> Dict[str, Any]:
    detector = Detector(integration_time=config.detector.get("integration_time", 1.0))
    pipeline = build_pipeline(config)
    processor = Processor(
        detector=detector,
        pipeline=pipeline,
        debug=config.debug.enabled,
        snapshot_dir=(run_dir / "snapshots") if config.exposure.snapshot else None,
    )

    metrics_all = []
    for step, t in enumerate(config.exposure.readout_times):
        logging.info("Exposure step %s at t=%.3f", step, t)
        detector.time = t  # type: ignore[attr-defined]
        processor.run_pipeline()
        metrics_all.append(processor.summarize())
    metrics = {"steps": metrics_all}
    dump_metrics(run_dir, metrics)
    return metrics


def run_batch(config: Configuration, config_path: Path, run_dir: Path) -> Dict[str, Any]:
    batch_results = []
    for idx, params in enumerate(config.batch.parameters):
        logging.info("Batch item %s", idx)
        overrides = {"detector": params.get("detector", {}), "pipeline": params.get("pipeline", {})}
        item_config = Configuration.from_yaml(config_path, overrides=overrides)
        sub_dir = run_dir / f"batch_{idx}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        res = run_exposure(item_config, sub_dir)
        batch_results.append(res)
    dump_metrics(run_dir, {"batch": batch_results})
    return {"batch": batch_results}


def main():
    parser = argparse.ArgumentParser(description="Run minimal detector simulation pipeline.")
    parser.add_argument("config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("--override", action="append", default=[], help="Override key=value (JSON values allowed)")
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    config = Configuration.from_yaml(args.config, overrides=overrides)

    run_dir = create_run_dir(config.output_root)
    dump_config(run_dir, {
        "config_file": str(Path(args.config).resolve()),
        "resolved": config_to_dict(config),
    })

    manifest = {
        "mode": config.mode,
        "stages": [stage for stage in ["scene", "photon", "charge", "readout", "postproc"]],
    }
    dump_manifest(run_dir, manifest)

    if config.mode == "exposure":
        run_exposure(config, run_dir)
    elif config.mode == "batch":
        run_batch(config, args.config, run_dir)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    logging.info("Run artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
