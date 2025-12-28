import os
from pathlib import Path

import pytest

from mini_framework.config import Configuration, build_pipeline
from mini_framework.contracts import Contract
from mini_framework.detector import Detector
from mini_framework.model_function import ModelFunction
from mini_framework.pipeline import DetectionPipeline
from mini_framework.processor import Processor
from mini_framework.registry import get as registry_get, register
from mini_framework.resolver import load_yaml_with_inheritance, resolve_placeholders, resolve_refs_in_obj
from mini_framework.stage import Stage


def test_env_and_placeholder_resolution(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
        detector:
          integration_time: 1.0
        pipeline:
          scene_generation:
            models: []
          photon_collection:
            models: []
          charge_generation:
            models: []
          charge_collection:
            models: []
          charge_measurement:
            models: []
          readout_electronics:
            models: []
          data_processing:
            models: []
        exposure:
          readout_times: [1]
        debug:
          dump_dir: ${env:DUMP_DIR:/tmp/default}
        gain: ${detector.integration_time}
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("DUMP_DIR", str(tmp_path / "logs"))
    raw = load_yaml_with_inheritance(cfg_path)
    raw = resolve_refs_in_obj(raw)
    resolved = resolve_placeholders(raw, raw)
    assert resolved["debug"]["dump_dir"] == str(tmp_path / "logs")
    assert resolved["gain"] == "1.0"


def test_inheritance_merge(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("detector: {integration_time: 1}\npipeline: {scene: {models: []}, photon:{models: []}, charge:{models: []}, readout:{models: []}, postproc:{models: []}}", encoding="utf-8")
    child.write_text("extends: base.yaml\ndetector: {integration_time: 2}\npipeline: {scene: {models: []}, photon:{models: []}, charge:{models: []}, readout:{models: []}, postproc:{models: []}}", encoding="utf-8")
    data = load_yaml_with_inheritance(child)
    assert data["detector"]["integration_time"] == 2


def test_contract_enforcement_raises():
    detector = Detector()
    stage = Stage("photon_collection", models=[], contract=Contract(requires=["scene"], produces=["photon"]))
    with pytest.raises(ValueError):
        stage.run(detector)


def test_registry_resolution():
    @register("_tmp_model")
    def _tmp_model(detector: Detector):
        detector.scene.data = 1  # type: ignore

    func = registry_get("_tmp_model")
    assert callable(func)


def test_end_to_end_exposure(tmp_path, monkeypatch):
    config_path = Path("examples/basic_imaging.yaml")
    monkeypatch.setenv("DARK_CURRENT", "0.3")
    config = Configuration.from_yaml(config_path)
    pipeline = build_pipeline(config)
    detector = Detector(
        rows=config.detector.get("rows", 64),
        cols=config.detector.get("cols", 64),
        integration_time=config.detector.get("integration_time", 1.0),
    )
    detector.output_dir = tmp_path
    processor = Processor(detector=detector, pipeline=pipeline, debug=False)
    processor.run_pipeline()
    assert detector.image.array is not None
    assert "stats" in detector.image.metadata


def test_smoke_run_outputs(tmp_path, monkeypatch):
    from run import run_exposure

    monkeypatch.setenv("DARK_CURRENT", "0.1")
    config = Configuration.from_yaml(
        Path("examples/basic_imaging.yaml"), overrides={"output_root": str(tmp_path)}
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_exposure(config, run_dir)

    images = list((run_dir / "images").glob("frame_*.npy"))
    assert images, "No image outputs written"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "stats.csv").exists() or (run_dir / "images").exists()
