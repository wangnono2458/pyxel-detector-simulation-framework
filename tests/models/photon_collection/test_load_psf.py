#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Tests for load psf model."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyxel.detectors import (
    CCD,
    CCDGeometry,
    Characteristics,
    Environment,
    ReadoutProperties,
)
from pyxel.models.photon_collection import load_psf


@pytest.fixture
def ccd_10x10_no_photons() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=10,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )

    detector._readout_properties = ReadoutProperties(times=[1.0])
    assert detector.photon.ndim == 0

    return detector


@pytest.fixture
def ccd_10x10_2d(ccd_10x10_no_photons: CCD) -> CCD:
    """Create a valid CCD detector."""
    detector = deepcopy(ccd_10x10_no_photons)
    assert detector.photon.ndim == 0
    detector.photon.array = np.full(fill_value=10.0, shape=detector.geometry.shape)

    detector._readout_properties = ReadoutProperties(times=[1.0])
    assert detector.photon.ndim == 2

    return detector


@pytest.fixture
def ccd_10x10_3d(ccd_10x10_no_photons: CCD) -> CCD:
    """Create a valid CCD detector."""
    detector = deepcopy(ccd_10x10_no_photons)
    assert detector.photon.ndim == 0

    num_wavelengths = 10
    num_rows, num_cols = detector.geometry.shape

    detector.photon.array_3d = xr.DataArray(
        np.full(
            fill_value=10.0,
            shape=(num_wavelengths, num_rows, num_cols),
            dtype=float,
        ),
        dims=["wavelength", "y", "x"],
        coords={
            "wavelength": np.linspace(start=300.0, stop=600.0, num=num_wavelengths)
        },
    )

    detector._readout_properties = ReadoutProperties(times=[1.0])
    assert detector.photon.ndim == 3

    return detector


@pytest.fixture(params=["CCD_2D", "CCD_3D"])
def ccd_detector(request, ccd_10x10_2d: CCD, ccd_10x10_3d: CCD) -> CCD:
    if request.param == "CCD_2D":
        assert ccd_10x10_2d.photon.ndim == 2
        return ccd_10x10_2d
    elif request.param == "CCD_3D":
        assert ccd_10x10_3d.photon.ndim == 3
        return ccd_10x10_3d
    else:
        raise NotImplementedError


@pytest.fixture
def psf_10x10() -> np.ndarray:
    """Create a gaussian psf of 10x10."""
    shape = (10, 10)
    ny, nx = (el / 2 for el in shape)
    ay, ax = (np.arange(-el / 2.0 + 0.5, el / 2.0 + 0.5) for el in shape)
    xx, yy = np.meshgrid(ax, ay, indexing="xy")
    r = ((xx / nx) ** 2 + (yy / ny) ** 2) ** 0.5
    out = np.ones(shape)
    sigma = 0.5
    out[...] = np.exp(-(r**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    return out


@pytest.fixture
def psf_10x10_npy_filename(psf_10x10: np.ndarray, tmp_path: Path) -> Path:
    """Create a filename with a psf 10x10."""
    filename = tmp_path / "psf.npy"
    np.save(filename, psf_10x10)
    return filename


@pytest.fixture
def psf_10x10_2d_npy_filename(psf_10x10: np.ndarray, tmp_path: Path) -> Path:
    filename = tmp_path / "psf_2d.npy"

    psf_3d = np.broadcast_to(psf_10x10, shape=(10, 10, 10))
    np.save(filename, psf_3d)
    return filename


@pytest.fixture
def bad_psf_10x10_npy_filename(tmp_path: Path, psf_10x10: np.ndarray) -> Path:
    filename = tmp_path / "bad_psf.npy"

    psf_10x10_4d = psf_10x10[np.newaxis, np.newaxis, :, :]
    assert psf_10x10_4d.ndim == 4

    np.save(filename, psf_10x10_4d)
    return filename


@pytest.fixture
def mismatch_psf_10x10_npy_filename(tmp_path: Path, psf_10x10: np.ndarray) -> Path:
    filename = tmp_path / "mismatch_psf.npy"

    psf_10x10_3d = psf_10x10[np.newaxis, :, :]

    np.save(filename, psf_10x10_3d)
    return filename


@pytest.mark.parametrize(
    "normalize_kernel",
    [
        pytest.param(True, id="With normalize_kernel"),
        pytest.param(False, id="Without normalize_kernel"),
    ],
)
def test_load_psf(
    ccd_detector: CCD,
    psf_10x10_npy_filename: Path,
    normalize_kernel: bool,
) -> None:
    """Test model 'load_psf'."""
    detector = ccd_detector

    # Run model
    load_psf(
        detector=detector,
        filename=psf_10x10_npy_filename,
        normalize_kernel=normalize_kernel,
    )


def test_load_psf_3d(ccd_10x10_3d: CCD, psf_10x10_2d_npy_filename: Path):
    """Test model 'load_psf'."""
    detector = ccd_10x10_3d

    # Run model
    load_psf(detector=detector, filename=psf_10x10_2d_npy_filename)


def test_load_psf_2d_psf_4d(ccd_10x10_3d: CCD, bad_psf_10x10_npy_filename: Path):
    """Test model 'load_psf' with bad inputs."""
    detector = ccd_10x10_3d

    with pytest.raises(ValueError, match=r"PSF kernel must be either 2D or 3D"):
        load_psf(detector=detector, filename=bad_psf_10x10_npy_filename)


def test_load_psf_2d_mismatch_psf(
    ccd_10x10_3d: CCD, mismatch_psf_10x10_npy_filename: Path
):
    """Test model 'load_psf' with bad inputs."""
    detector = ccd_10x10_3d

    with pytest.raises(ValueError, match=r"Mismatch with the number of wavelengths"):
        load_psf(detector=detector, filename=mismatch_psf_10x10_npy_filename)


def test_load_psf_no_photons(ccd_10x10_no_photons: CCD, psf_10x10_npy_filename: Path):
    """Test model 'load_psf' with bad inputs."""
    detector = ccd_10x10_no_photons

    with pytest.raises(ValueError, match=r"Photon array must be 2D or 3D"):
        load_psf(detector=detector, filename=psf_10x10_npy_filename)
