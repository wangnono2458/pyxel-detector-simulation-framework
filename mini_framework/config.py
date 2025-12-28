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
    scene_generation: StageSpec = field(default_factory=StageSpec)
    photon_collection: StageSpec = field(default_factory=StageSpec)
    charge_generation: StageSpec = field(default_factory=StageSpec)
    charge_collection: StageSpec = field(default_factory=StageSpec)
    charge_transfer: StageSpec = field(default_factory=StageSpec)
    charge_measurement: StageSpec = field(default_factory=StageSpec)
    readout_electronics: StageSpec = field(default_factory=StageSpec)
    data_processing: StageSpec = field(default_factory=StageSpec)
    phasing: StageSpec = field(default_factory=StageSpec)
    signal_transfer: StageSpec = field(default_factory=StageSpec)


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
    output_root: Path = Path("./output")

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
        scene_generation=_parse_stage(pipeline_data.get("scene_generation", {})),
        photon_collection=_parse_stage(pipeline_data.get("photon_collection", {})),
        charge_generation=_parse_stage(pipeline_data.get("charge_generation", {})),
        charge_collection=_parse_stage(pipeline_data.get("charge_collection", {})),
        charge_transfer=_parse_stage(pipeline_data.get("charge_transfer", {})),
        charge_measurement=_parse_stage(pipeline_data.get("charge_measurement", {})),
        readout_electronics=_parse_stage(pipeline_data.get("readout_electronics", {})),
        data_processing=_parse_stage(pipeline_data.get("data_processing", {})),
        phasing=_parse_stage(pipeline_data.get("phasing", {})),
        signal_transfer=_parse_stage(pipeline_data.get("signal_transfer", {})),
    )


def build_pipeline(config: Configuration):
    from mini_framework.pipeline import DetectionPipeline
    from mini_framework.model_function import ModelFunction
    from mini_framework.contracts import Contract
    from mini_framework.stage import Stage

    def to_stage(name: str, spec: StageSpec):
        models = [ModelFunction(**m.__dict__) for m in spec.models]
        contract = Contract(spec.contract_requires, spec.contract_produces)
        return Stage(name=name, models=models, contract=contract)

    stage_map = {
        "scene_generation": to_stage("scene_generation", config.pipeline.scene_generation),
        "photon_collection": to_stage("photon_collection", config.pipeline.photon_collection),
        "phasing": to_stage("phasing", config.pipeline.phasing),
        "charge_generation": to_stage("charge_generation", config.pipeline.charge_generation),
        "charge_collection": to_stage("charge_collection", config.pipeline.charge_collection),
        "charge_transfer": to_stage("charge_transfer", config.pipeline.charge_transfer),
        "charge_measurement": to_stage("charge_measurement", config.pipeline.charge_measurement),
        "signal_transfer": to_stage("signal_transfer", config.pipeline.signal_transfer),
        "readout_electronics": to_stage("readout_electronics", config.pipeline.readout_electronics),
        "data_processing": to_stage("data_processing", config.pipeline.data_processing),
    }

    return DetectionPipeline.from_stage_map(stage_map)


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
            "scene_generation": stage_to_dict(config.pipeline.scene_generation),
            "photon_collection": stage_to_dict(config.pipeline.photon_collection),
            "phasing": stage_to_dict(config.pipeline.phasing),
            "charge_generation": stage_to_dict(config.pipeline.charge_generation),
            "charge_collection": stage_to_dict(config.pipeline.charge_collection),
            "charge_transfer": stage_to_dict(config.pipeline.charge_transfer),
            "charge_measurement": stage_to_dict(config.pipeline.charge_measurement),
            "signal_transfer": stage_to_dict(config.pipeline.signal_transfer),
            "readout_electronics": stage_to_dict(config.pipeline.readout_electronics),
            "data_processing": stage_to_dict(config.pipeline.data_processing),
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
