from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from mini_framework import buckets
from mini_framework.detector import Detector
from mini_framework.outputs import dump_snapshot
from mini_framework.pipeline import DetectionPipeline


class Processor:
    def __init__(
        self,
        detector: Detector,
        pipeline: DetectionPipeline,
        *,
        debug: bool = False,
        snapshot_dir: Optional[Path] = None,
    ):
        self.detector = detector
        self.pipeline = pipeline
        self.debug = debug
        self.snapshot_dir = snapshot_dir
        self._log = logging.getLogger(__name__)

        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self) -> None:
        self._log.info("Starting pipeline")
        for stage in self.pipeline:
            stage.run(self.detector, debug=self.debug)
            if self.snapshot_dir and self.debug:
                dump_snapshot(self.snapshot_dir, stage.name, self.detector)

    def summarize(self) -> Dict[str, dict]:
        return {
            "scene": buckets.summarize_bucket(self.detector.scene),
            "photon": buckets.summarize_bucket(self.detector.photon),
            "charge": buckets.summarize_bucket(self.detector.charge),
            "image": buckets.summarize_bucket(self.detector.image),
        }
