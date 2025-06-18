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
from PIL import Image

from pyxel.inputs import load_image_v2

# This is equivalent to 'import pytest_httpserver'
pytest_httpserver = pytest.importorskip(
    "pytest_httpserver",
    reason="Package 'pytest_httpserver' is not installed. Use 'pip install pytest-httpserver'",
)


@pytest.fixture(
    params=[
        pytest.param(("file", "image.fits"), id="fits"),
        pytest.param(("file", "image.FITS"), id="FITS"),
        pytest.param(("file", "image_hdus.fits"), id="HDUs + fits"),
        pytest.param(("file", "image.npy"), id="npy"),
        pytest.param(("file", "image_tab.txt"), id="txt + tab"),
        pytest.param(("file", "image_space.txt"), id="txt + space"),
        pytest.param(("file", "image_comma.txt"), id="txt + comma"),
        pytest.param(("file", "image_pipe.txt"), id="txt + pipe"),
        pytest.param(("file", "image_semicolon.txt"), id="txt + semicolon"),
        pytest.param(("url", "image.fits"), id="URL + fits"),
    ]
)
def image_filename(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    httpserver: pytest_httpserver.HTTPServer,
) -> str | Path:
    """Build an image."""
    param: tuple[str, str] = request.param
    file_type, file_name = param

    data_2d = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    if file_type == "file":
        full_path: Path = tmp_path / file_name

        if file_name.lower() == "image.fits":
            fits.writeto(full_path, data=data_2d)

        elif file_name.lower() == "image_hdus.fits":
            hdu_primary = fits.PrimaryHDU(data_2d)
            hdu_secondary = fits.ImageHDU(data_2d * 2)
            hdu_lst = fits.HDUList([hdu_primary, hdu_secondary])

            hdu_lst.writeto(full_path)

        elif file_name.lower() == "image.npy":
            np.save(full_path, arr=data_2d)
        elif file_name.lower() == "image_tab.txt":
            np.savetxt(full_path, X=data_2d, delimiter="\t")
        elif file_name.lower() == "image_space.txt":
            np.savetxt(full_path, X=data_2d, delimiter=" ")
        elif file_name.lower() == "image_comma.txt":
            np.savetxt(full_path, X=data_2d, delimiter=",")
        elif file_name.lower() == "image_pipe.txt":
            np.savetxt(full_path, X=data_2d, delimiter="|")
        elif file_name.lower() == "image_semicolon.txt":
            np.savetxt(full_path, X=data_2d, delimiter=";")
        elif file_name.lower() == "image.png":
            pil_image: Image = Image.fromarray(data_2d)
            pil_image.save(full_path)

        else:
            raise NotImplementedError

        return full_path

    elif file_type == "url":
        # Extract an url (e.g. 'http://localhost:59226/)
        full_url: str = httpserver.url_for(file_name)

        tmp_folder: Path = tmp_path / "tmp"
        tmp_folder.mkdir(parents=True, exist_ok=True)

        if file_name.lower() == "image.fits":
            content_type: str = "text/plain"
            tmp_filename = tmp_folder / file_name
            fits.writeto(tmp_filename, data=data_2d)
        else:
            raise NotImplementedError

        response_data_bytes: bytes = Path(tmp_filename).read_bytes()
        httpserver.expect_request(f"/{file_name}").respond_with_data(
            response_data_bytes, content_type=content_type
        )
        return full_url

    else:
        raise NotImplementedError


def test_load_image_data_array(image_filename: str | Path):
    """Test function 'load_image_v2'."""
    data_array: xr.DataArray = load_image_v2(image_filename, rename_dims={})
    assert isinstance(data_array, xr.DataArray)

    exp_data_array = xr.DataArray(
        np.array([[1, 2], [3, 4]], dtype=np.uint16),
        dims=["y", "x"],
    )
    xr.testing.assert_equal(data_array, exp_data_array)
