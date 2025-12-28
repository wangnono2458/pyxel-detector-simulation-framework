# Minimal Pyxel-like Imaging Pipeline

本框架复刻 Pyxel 的核心体验（配置驱动 + 固定阶段 + 模型插件），但实现方式全新、轻量，可直接运行红外成像最小链路。

## 固定阶段 & 数据桶

阶段顺序（可为空）：
`scene_generation -> photon_collection -> phasing -> charge_generation -> charge_collection -> charge_transfer -> charge_measurement -> signal_transfer -> readout_electronics -> data_processing`

Detector 持有桶：`Scene`（scene.data）、`Photon`（photon.photons）、`Charge`（charge.electrons）、`Pixel`（pixel.signal）、`Image`（image.array）。Contract 在阶段运行前/后校验必需桶是否存在并被填充。

## 配置（唯一入口）

YAML 字段：
- `mode: exposure`
- `detector`: `rows`, `cols`, `integration_time`
- `exposure`: `readout_times`
- `pipeline`: 每个阶段 `requires/produces/models[]`，模型字段 `name/func/enabled/arguments`

支持 `${env:VAR}`、`${path}` 占位解析（`detector.output_dir` 等可被引用）。

示例：`examples/basic_imaging.yaml`
```yaml
pipeline:
  scene_generation:
    requires: []
    produces: [scene]
    models:
      - name: stars
        func: point_sources
        enabled: true
        arguments:
          background: 5.0
          sources:
            - {x: 10, y: 10, flux: 200.0}
  photon_collection:
    requires: [scene]
    produces: [photon]
    models:
      - name: psf
        func: photon_psf_throughput
        enabled: true
        arguments:
          psf_sigma: 1.2
          throughput: 0.85
```

## 运行

```bash
python run.py examples/basic_imaging.yaml
```
输出写入 `./output/<run_id>/`：`images/frame_*.npy/json`、`metrics.json`、`stats.csv`（由 data_processing 生成），以及配置/清单。

## 内置最小物理模型
- scene_generation: `point_sources`（背景 + 点源）
- photon_collection: `photon_psf_throughput`（高斯 PSF + 通量/波段缩放）
- charge_generation: `qe_shotnoise`（QE + 泊松噪声）
- charge_collection: `add_dark_current`（按积分时间加暗电流）
- charge_measurement: `read_noise_and_gain`（读噪 + 增益 + 偏置，输出 Pixel 信号）
- readout_electronics: `quantize_clip`（量化并裁剪到位深）
- data_processing: `compute_stats`（全局/ROI 均值方差，CSV 输出）

## 扩展模型
实现函数 `def my_model(detector: Detector, **kwargs)` 并用 `@register("my_model")` 装饰，在 YAML 中 `func: my_model` 即可。
