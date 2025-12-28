from __future__ import annotations

import numpy as np

from mini_framework.detector import Detector
from mini_framework.registry import register


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    out = np.zeros_like(image, dtype=float)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            out[i, j] = np.sum(region * kernel)
    return out


@register("photon_psf_throughput")
def photon_psf_throughput(
    detector: Detector,
    psf_sigma: float = 1.0,
    throughput: float = 0.8,
    band_factor: float = 1.0,
):
    if detector.scene.data is None:
        raise ValueError("Scene data missing; run scene_generation first")
    kernel_size = int(max(3, psf_sigma * 6))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = _gaussian_kernel(kernel_size, psf_sigma)
    blurred = _convolve2d(detector.scene.data, kernel)
    photons = blurred * throughput * band_factor
    detector.photon.photons = photons
    detector.photon.metadata.update({"psf_sigma": psf_sigma, "throughput": throughput, "band_factor": band_factor})
