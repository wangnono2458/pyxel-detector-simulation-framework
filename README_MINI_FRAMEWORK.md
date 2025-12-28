# Minimal Infrared Detector Simulation Framework

This folder contains a small, configuration-driven pipeline inspired by Pyxel. It keeps the *methodology* (config + fixed stage order + pluggable models) but uses a lightweight, registry-based implementation.

## Stage order & contracts

Fixed stage order: `scene -> photon -> charge -> readout -> postproc`.

Each stage carries a **contract**:
- `requires`: buckets that must be populated before the stage (e.g., `scene`, `photon`, `charge`, `image`).
- `produces`: buckets that must be populated after the stage.

If a contract is not satisfied, the run fails early.

## Configuration system

- YAML with inheritance via `extends: base.yaml` (relative to the file).
- Supports `${env:VAR}` (with optional default) and `${path.to.value}` placeholders resolved from the loaded configuration.
- `func` can be either a registered model name (preferred) or `module:function` for dynamic import.
- Command-line overrides: `--override pipeline.readout.models.0.arguments.gain=1.2` (numbers/JSON are auto-parsed when possible).

### Example: `examples/ir_pipeline.yaml`
```yaml
extends: ./base.yaml
mode: exposure
pipeline:
  scene:
    requires: []
    produces: [scene]
    models:
      - name: background
        func: ir_background
        enabled: true
        arguments:
          sky_level: 20.0
  # ...
```

## Running

```bash
python run.py examples/ir_pipeline.yaml --override detector.integration_time=3.0
```

Artifacts are written under `runs/<run_id>/`:
- `config.resolved.yaml` – resolved configuration
- `manifest.json` – mode and stage info
- `metrics.json` – per-step summaries
- `snapshots/` – optional per-stage summaries (when snapshots are enabled)

## Adding a new model

1. Implement a function `def my_model(detector: Detector, **kwargs): ...`.
2. Register it:
   ```python
   from mini_framework.registry import register

   @register("my_model")
   def my_model(detector: Detector, factor: float = 1.0):
       ...
   ```
3. Reference it in YAML:
   ```yaml
   func: my_model
   arguments:
     factor: 2.0
   ```

## Modes

- **exposure**: iterate over `exposure.readout_times`, running the pipeline each step.
- **batch**: scan parameter sets in `batch.parameters`; each item may override `detector`/`pipeline` keys.

## Tests

Run the focused test suite:
```bash
python -m pytest tests/test_mini_framework.py
```
