#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from astropy.io import fits

from pyxel.detectors import (
    CCD,
    CCDGeometry,
    Characteristics,
    Environment,
    ReadoutProperties,
)
from pyxel.models.photon_collection import load_wavelength_psf


@pytest.fixture
def num_wavelengths() -> int:
    return 5


@pytest.fixture
def ccd_10x10_3d(num_wavelengths: int) -> CCD:
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


@pytest.fixture
def psf_10x10_3d(num_wavelengths: int) -> xr.DataArray:
    """Create a gaussian psf of 10x10."""
    shape = (10, 10)
    ny, nx = (el / 2 for el in shape)
    ay, ax = (np.arange(-el / 2.0 + 0.5, el / 2.0 + 0.5) for el in shape)
    xx, yy = np.meshgrid(ax, ay, indexing="xy")
    r = ((xx / nx) ** 2 + (yy / ny) ** 2) ** 0.5
    psf_2d = np.ones(shape)
    sigma = 0.5
    psf_2d[...] = np.exp(-(r**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

    wavelength_1d = np.linspace(start=1.0, stop=2.0, num=num_wavelengths)
    wavelength_3d = wavelength_1d[:, np.newaxis, np.newaxis]

    psf_3d = psf_2d[np.newaxis, :, :] * wavelength_3d
    assert psf_3d.shape == (num_wavelengths, 10, 10)

    return xr.DataArray(
        psf_3d,
        dims=["wavelength", "y", "x"],
        coords={
            "wavelength": np.linspace(start=200.0, stop=700.0, num=num_wavelengths)
        },
    )


@pytest.fixture
def psf_filename(tmp_path: Path, psf_10x10_3d: xr.DataArray) -> Path:
    assert psf_10x10_3d.ndim == 3

    filename = tmp_path / "psf_3d.fits"

    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(np.asarray(psf_10x10_3d)),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="waves",
                        array=np.asarray(psf_10x10_3d["wavelength"]),
                        format="K",
                    )
                ]
            ),
        ]
    )
    hdus.writeto(filename)

    return filename


def test_load_wavelength_psf(ccd_10x10_3d: CCD, psf_filename: Path):
    """Test model 'load_wavelength_psf'."""
    detector = ccd_10x10_3d

    load_wavelength_psf(
        detector,
        filename=psf_filename,
        wavelength_col="dim_0",
        x_col="dim_2",
        y_col="dim_1",
        wavelength_table_name="waves",
        normalize_kernel=False,
    )
