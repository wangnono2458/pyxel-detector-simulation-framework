from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mini_framework.resolver import (
    load_yaml_with_inheritance,
    merge_dicts,
    resolve_placeholders,
    resolve_refs_in_obj,
)


@dataclass
class ModelSpec:
    name: str
    func: str
    enabled: bool = True
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageSpec:
    models: List[ModelSpec] = field(default_factory=list)
    contract_requires: List[str] = field(default_factory=list)
    contract_produces: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    scene: StageSpec = field(default_factory=StageSpec)
    photon: StageSpec = field(default_factory=StageSpec)
    charge: StageSpec = field(default_factory=StageSpec)
    readout: StageSpec = field(default_factory=StageSpec)
    postproc: StageSpec = field(default_factory=StageSpec)


@dataclass
class ExposureConfig:
    readout_times: List[float] = field(default_factory=lambda: [1.0])
    snapshot: bool = False


@dataclass
class BatchConfig:
    parameters: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DebugConfig:
    enabled: bool = False
    dump_dir: Optional[Path] = None

    def __post_init__(self):
        if self.dump_dir is not None:
            self.dump_dir = Path(self.dump_dir).expanduser().resolve()


@dataclass
class Configuration:
    detector: Dict[str, Any]
    pipeline: PipelineConfig
    mode: str = "exposure"  # exposure | batch
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    output_root: Path = Path("./runs")

    def __post_init__(self):
        self.output_root = Path(self.output_root).expanduser().resolve()

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> "Configuration":
        data = load_yaml_with_inheritance(Path(path))
        if overrides:
            data = merge_dicts(data, overrides)

        data = resolve_refs_in_obj(data)

        if "detector" not in data or "pipeline" not in data:
            raise ValueError("Configuration YAML must define 'detector' and 'pipeline'.")

        data = resolve_placeholders(data, data)

        pipeline = _parse_pipeline(data["pipeline"])
        exposure = ExposureConfig(**data.get("exposure", {}))
        batch = BatchConfig(**data.get("batch", {}))
        debug = DebugConfig(**data.get("debug", {}))

        return cls(
            detector=data["detector"],
            pipeline=pipeline,
            mode=data.get("mode", "exposure"),
            exposure=exposure,
            batch=batch,
            debug=debug,
            output_root=Path(data.get("output_root", "./runs")),
        )


def _parse_stage(stage_data: Dict[str, Any]) -> StageSpec:
    models_data = stage_data.get("models", stage_data if isinstance(stage_data, list) else stage_data.get("models", []))
    models: List[ModelSpec] = []
    for item in models_data or []:
        models.append(ModelSpec(**item))
    return StageSpec(
        models=models,
        contract_requires=stage_data.get("requires", []),
        contract_produces=stage_data.get("produces", []),
    )


def _parse_pipeline(pipeline_data: Dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
        scene=_parse_stage(pipeline_data.get("scene", {})),
        photon=_parse_stage(pipeline_data.get("photon", {})),
        charge=_parse_stage(pipeline_data.get("charge", {})),
        readout=_parse_stage(pipeline_data.get("readout", {})),
        postproc=_parse_stage(pipeline_data.get("postproc", {})),
    )


def build_pipeline(config: Configuration):
    from mini_framework.pipeline import DetectionPipeline
    from mini_framework.model_function import ModelFunction
    from mini_framework.contract import Contract
    from mini_framework.stage import Stage

    def to_stage(name: str, spec: StageSpec):
        models = [ModelFunction(**m.__dict__) for m in spec.models]
        contract = Contract(spec.contract_requires, spec.contract_produces)
        return Stage(name=name, models=models, contract=contract)

    return DetectionPipeline(
        stages=[
            to_stage("scene", config.pipeline.scene),
            to_stage("photon", config.pipeline.photon),
            to_stage("charge", config.pipeline.charge),
            to_stage("readout", config.pipeline.readout),
            to_stage("postproc", config.pipeline.postproc),
        ]
    )


def stage_to_dict(stage: StageSpec) -> Dict[str, Any]:
    return {
        "requires": stage.contract_requires,
        "produces": stage.contract_produces,
        "models": [m.__dict__ for m in stage.models],
    }


def config_to_dict(config: Configuration) -> Dict[str, Any]:
    return {
        "mode": config.mode,
        "detector": config.detector,
        "pipeline": {
            "scene": stage_to_dict(config.pipeline.scene),
            "photon": stage_to_dict(config.pipeline.photon),
            "charge": stage_to_dict(config.pipeline.charge),
            "readout": stage_to_dict(config.pipeline.readout),
            "postproc": stage_to_dict(config.pipeline.postproc),
        },
        "exposure": {
            "readout_times": config.exposure.readout_times,
            "snapshot": config.exposure.snapshot,
        },
        "batch": {"parameters": config.batch.parameters},
        "debug": {
            "enabled": config.debug.enabled,
            "dump_dir": str(config.debug.dump_dir) if config.debug.dump_dir else None,
        },
        "output_root": str(config.output_root),
    }
