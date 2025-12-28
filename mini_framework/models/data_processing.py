from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


@register("compute_stats")
def compute_stats(detector: Detector, output_dir: str | None = "./output", roi: list[int] | None = None):
    if detector.image.array is None:
        raise ValueError("Image missing; run readout first")
    arr = detector.image.array.astype(float)
    stats = {
        "global_mean": float(np.mean(arr)),
        "global_var": float(np.var(arr)),
    }
    if roi and len(roi) == 4:
        y0, y1, x0, x1 = roi
        sub = arr[y0:y1, x0:x1]
        stats.update({
            "roi_mean": float(np.mean(sub)),
            "roi_var": float(np.var(sub)),
        })
    dest = detector.output_dir if output_dir in (None, "") else output_dir
    outdir = Path(dest if dest is not None else "./output").expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for k, v in stats.items():
            writer.writerow([k, v])
    detector.image.metadata["stats"] = stats
