#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import os
import re
from pathlib import Path
from typing import no_type_check

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

import pyxel
from pyxel import load_header, load_image

# This is equivalent to 'import pytest_httpserver'
pytest_httpserver = pytest.importorskip(
    "pytest_httpserver",
    reason="Package 'pytest_httpserver' is not installed. Use 'pip install pytest-httpserver'",
)


@pytest.fixture
def valid_multiple_hdus() -> fits.HDUList:
    """Create a valid HDUList with one 'PrimaryHDU' and one 'ImageHDU'."""
    # Create a new image
    primary_2d = np.array([[5, 6], [7, 8]], dtype=np.uint16)
    secondary_2d = np.array([[9, 10], [11, 12]], dtype=np.uint16)

    hdu_primary = fits.PrimaryHDU(primary_2d, header=fits.Header({"hello": "world"}))
    hdu_secondary = fits.ImageHDU(
        secondary_2d, header=fits.Header({"foo": "bar"}), name="OTHER_IMAGE"
    )
    hdu_lst = fits.HDUList([hdu_primary, hdu_secondary])

    return hdu_lst


@pytest.fixture
def valid_pil_image() -> Image.Image:
    """Create a valid RGB PIL image."""
    data_2d = np.array([[10, 20], [30, 40]], dtype="uint8")
    pil_image = Image.fromarray(data_2d).convert("RGB")

    return pil_image


@no_type_check
@pytest.fixture
def valid_data2d_http_hostname(
    tmp_path: Path,
    httpserver: pytest_httpserver.HTTPServer,
    valid_pil_image: Image.Image,
    valid_multiple_hdus: fits.HDUList,
) -> str:
    """Create valid 2D files on a temporary folder and HTTP server."""
    # Get current folder
    current_folder: Path = Path().cwd()

    try:
        os.chdir(tmp_path)

        # Create folder 'data'
        Path("data").mkdir(parents=True, exist_ok=True)

        data_2d = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        # Save 2d images
        fits.writeto("data/img.fits", data=data_2d)
        fits.writeto("data/img2.FITS", data=data_2d)
        valid_multiple_hdus.writeto("data/img_multiple.fits")
        valid_multiple_hdus.writeto("data/img_multiple2.FITS")
        np.save("data/img.npy", arr=data_2d)
        np.savetxt("data/img_tab.txt", X=data_2d, delimiter="\t")
        np.savetxt("data/img_space.txt", X=data_2d, delimiter=" ")
        np.savetxt("data/img_comma.txt", X=data_2d, delimiter=",")
        np.savetxt("data/img_pipe.txt", X=data_2d, delimiter="|")
        np.savetxt("data/img_semicolon.txt", X=data_2d, delimiter=";")

        valid_pil_image.save("data/img.jpg")
        valid_pil_image.save("data/img.jpeg")
        valid_pil_image.save("data/img.png")
        valid_pil_image.save("data/img2.PNG")
        valid_pil_image.save("data/img.tiff")
        valid_pil_image.save("data/img.tif")
        valid_pil_image.save("data/img.bmp")

        text_filenames = [
            "data/img_tab.txt",
            "data/img_space.txt",
            "data/img_comma.txt",
            "data/img_pipe.txt",
            "data/img_semicolon.txt",
        ]

        # Put text data in a fake HTTP server
        for filename in text_filenames:
            response_data: str = Path(filename).read_text()
            httpserver.expect_request(f"/{filename}").respond_with_data(
                response_data, content_type="text/plain"
            )

        binary_filenames = [
            ("data/img.fits", "text/plain"),  # TODO: Change this type
            ("data/img2.FITS", "text/plain"),  # TODO: Change this type
            ("data/img_multiple.fits", "text/plain"),  # TODO: Change this type
            ("data/img_multiple2.FITS", "text/plain"),  # TODO: Change this type
            ("data/img.npy", "text/plain"),  # TODO: Change this type
            ("data/img.jpg", "image/jpeg"),
            ("data/img.jpeg", "image/jpeg"),
            ("data/img.png", "image/png"),
            ("data/img2.PNG", "image/png"),
            ("data/img.tif", "image/tiff"),
            ("data/img.tiff", "image/tiff"),
            ("data/img.bmp", "image/bmp"),
        ]

        # Put binary data in a fake HTTP server
        for filename, content_type in binary_filenames:
            response_data_bytes: bytes = Path(filename).read_bytes()
            httpserver.expect_request(f"/{filename}").respond_with_data(
                response_data_bytes, content_type=content_type
            )

        # Extract an url (e.g. 'http://localhost:59226/)
        url: str = httpserver.url_for("")

        # Extract the hostname (e.g. 'localhost:59226')
        hostname: str = re.findall(r"http://(.*)/", url)[0]

        yield hostname

    finally:
        os.chdir(current_folder)


@pytest.fixture
def data_2d_multi_hdus(tmp_path: Path) -> Path:
    """Generate a 2D fits file with multiple HDUs."""
    filename = tmp_path / "data_multi_hdus.fits"

    primary_2d = np.array([[5, 6], [7, 8]], dtype=np.uint16)
    secondary_2d = np.array([[9, 10], [11, 12]], dtype=np.uint16)

    hdu_primary = fits.PrimaryHDU(primary_2d, header=fits.Header({"hello": "world"}))
    hdu_secondary = fits.ImageHDU(
        secondary_2d, header=fits.Header({"foo": "bar"}), name="OTHER_IMAGE"
    )

    hdus = fits.HDUList([hdu_primary, hdu_secondary])
    hdus.writeto(filename)

    return filename


@pytest.fixture
def invalid_data2d_hostname(
    tmp_path: Path, httpserver: pytest_httpserver.HTTPServer
) -> str:
    """Create invalid 2D files on a temporary folder and HTTP server."""
    # Get current folder
    current_folder: Path = Path().cwd()

    try:
        os.chdir(tmp_path)

        # Create folder 'data'
        Path("invalid_data").mkdir(parents=True, exist_ok=True)

        data_2d = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        np.savetxt("invalid_data/img_X.txt", X=data_2d, delimiter="X")

        text_filenames = ["invalid_data/img_X.txt"]

        # Put text data in a fake HTTP server
        for filename in text_filenames:
            response_data: str = Path(filename).read_text()
            httpserver.expect_request(f"/{filename}").respond_with_data(
                response_data, content_type="text/plain"
            )

        # Extract an url (e.g. 'http://localhost:59226/)
        url: str = httpserver.url_for("")

        # Extract the hostname (e.g. 'localhost:59226')
        hostname: str = re.findall(r"http://(.*)/", url)[0]

        yield hostname

    finally:
        os.chdir(current_folder)


@pytest.mark.parametrize(
    "filename, exp_error, exp_message",
    [
        ("dummy", ValueError, "Image format not supported"),
        ("dummy.foo", ValueError, "Image format not supported"),
        ("unknown.fits", FileNotFoundError, None),
        (Path("unknown.fits"), FileNotFoundError, r"can not be found"),
        ("https://domain/unknown.fits", FileNotFoundError, None),
        ("invalid_data/img_X.txt", ValueError, "Cannot find the separator"),
        (
            "http://{host}/invalid_data/img_X.txt",
            ValueError,
            "Cannot find the separator",
        ),
    ],
)
def test_invalid_filename(
    invalid_data2d_hostname: str,
    filename,
    exp_error: TypeError,
    exp_message: str | None,
):
    """Test invalid filenames."""
    if isinstance(filename, str):
        new_filename: str = filename.format(host=invalid_data2d_hostname)
    else:
        new_filename = filename

    with pytest.raises(exp_error, match=exp_message):
        _ = load_image(new_filename)


@pytest.mark.skip(reason="Fix these tests")
@pytest.mark.parametrize("with_caching", [False, True])
@pytest.mark.parametrize(
    "filename, exp_data",
    [
        # FITS files
        ("data/img.fits", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img2.FITS", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img_multiple.fits", np.array([[5, 6], [7, 8]], np.uint16)),
        ("data/img_multiple2.FITS", np.array([[5, 6], [7, 8]], np.uint16)),
        (Path("data/img.fits"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("./data/img.fits"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_multiple.fits"), np.array([[5, 6], [7, 8]], np.uint16)),
        ("http://{host}/data/img.fits", np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_multiple.fits", np.array([[5, 6], [7, 8]], np.uint16)),
        (
            "http://{host}/data/img_multiple2.FITS",
            np.array([[5, 6], [7, 8]], np.uint16),
        ),
        # Numpy binary files
        ("data/img.npy", np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img.npy"), np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img.npy", np.array([[1, 2], [3, 4]], np.uint16)),
        # Numpy text files
        ("data/img_tab.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img_space.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img_comma.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img_pipe.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("data/img_semicolon.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_tab.txt"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_space.txt"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_comma.txt"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_pipe.txt"), np.array([[1, 2], [3, 4]], np.uint16)),
        (Path("data/img_semicolon.txt"), np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_tab.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_space.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_comma.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_pipe.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        ("http://{host}/data/img_semicolon.txt", np.array([[1, 2], [3, 4]], np.uint16)),
        # JPG files
        ("data/img.jpg", np.array([[13, 19], [28, 34]])),
        ("data/img.jpeg", np.array([[13, 19], [28, 34]])),
        ("http://{host}/data/img.jpg", np.array([[13, 19], [28, 34]])),
        ("http://{host}/data/img.jpeg", np.array([[13, 19], [28, 34]])),
        # PNG files
        ("data/img.png", np.array([[10, 20], [30, 40]])),
        ("data/img2.PNG", np.array([[10, 20], [30, 40]])),
        ("http://{host}/data/img.png", np.array([[10, 20], [30, 40]])),
        ("http://{host}/data/img2.PNG", np.array([[10, 20], [30, 40]])),
        # TIFF files
        ("data/img.tif", np.array([[10, 20], [30, 40]])),
        ("data/img.tiff", np.array([[10, 20], [30, 40]])),
        ("http://{host}/data/img.tif", np.array([[10, 20], [30, 40]])),
        ("http://{host}/data/img.tiff", np.array([[10, 20], [30, 40]])),
        # BMP files
        ("data/img.bmp", np.array([[10, 20], [30, 40]])),
    ],
)
def test_load_image(
    with_caching: bool,
    valid_data2d_http_hostname: str,
    filename: str | Path,
    exp_data: np.ndarray,
):
    """Test function 'load_image' with local and remote files."""
    with pyxel.set_options(cache_enabled=with_caching):
        if isinstance(filename, Path):
            # Load data
            data_2d = load_image(filename)
        else:
            full_url: str = filename.format(host=valid_data2d_http_hostname)

            # Load data
            data_2d = load_image(full_url)

        # Check 'data_2d
        np.testing.assert_equal(data_2d, exp_data)


def test_load_image_with_data_path(data_2d_multi_hdus: Path):
    """Test function 'load_image' with parameter 'data_path'."""
    filename: Path = data_2d_multi_hdus
    assert filename.exists()

    exp_primary_2d = np.array([[5, 6], [7, 8]], dtype=np.uint16)
    exp_secondary_2d = np.array([[9, 10], [11, 12]], dtype=np.uint16)

    # Load data with 'data_path' param
    data_2d = load_image(filename)
    np.testing.assert_equal(data_2d, exp_primary_2d)

    # Load data with 'data_path' as an int
    data_2d = load_image(filename, data_path=0)
    np.testing.assert_equal(data_2d, exp_primary_2d)

    data_2d = load_image(filename, data_path=1)
    np.testing.assert_equal(data_2d, exp_secondary_2d)

    with pytest.raises(OSError, match="Cannot access"):
        _ = load_image(filename, data_path=2)

    # Load data with 'data_path' as a str
    data_2d = load_image(filename, data_path="PRIMARY")
    np.testing.assert_equal(data_2d, exp_primary_2d)

    data_2d = load_image(filename, data_path="OTHER_IMAGE")
    np.testing.assert_equal(data_2d, exp_secondary_2d)

    with pytest.raises(OSError, match="Cannot access"):
        _ = load_image(filename, data_path="NOT_EXISTING")


@pytest.mark.skip(reason="Fix these tests")
@pytest.mark.parametrize("with_caching", [False, True])
@pytest.mark.parametrize(
    "filename, section, exp_header",
    [
        (
            "data/img.fits",
            None,
            {"BSCALE": 1, "BZERO": 32768, "EXTEND": True},
        ),
        (
            "http://{host}/data/img.fits",
            None,
            {
                "BITPIX": (16, "array data type"),
                "BSCALE": (1, ""),
                "BZERO": (32768, ""),
                "EXTEND": (True, ""),
                "NAXIS": (2, "number of array dimensions"),
                "NAXIS1": (2, ""),
                "NAXIS2": (2, ""),
                "SIMPLE": (True, "conforms to FITS standard"),
            },
        ),
        ("data/img.jpg", None, None),
        (
            "data/img_multiple.fits",
            None,
            {
                "BITPIX": (16, "array data type"),
                "BSCALE": (1, ""),
                "BZERO": (32768, ""),
                "EXTEND": (True, ""),
                "HELLO": ("world", ""),  # Specific for first ext
                "NAXIS": (2, "number of array dimensions"),
                "NAXIS1": (2, ""),
                "NAXIS2": (2, ""),
                "SIMPLE": (True, "conforms to FITS standard"),
            },
        ),
        (
            "data/img_multiple.fits",
            "OTHER_IMAGE",
            {
                "BITPIX": (16, "array data type"),
                "BSCALE": (1, ""),
                "BZERO": (32768, ""),
                "EXTNAME": ("OTHER_IMAGE", "extension name"),  # for second ext
                "FOO": ("bar", ""),
                "GCOUNT": (1, "number of groups"),
                "NAXIS": (2, "number of array dimensions"),
                "NAXIS1": (2, ""),
                "NAXIS2": (2, ""),
                "PCOUNT": (0, "number of parameters"),
                "XTENSION": ("IMAGE", "Image extension"),
            },
        ),
        (
            "http://{host}/data/img_multiple.fits",
            None,
            {
                "BITPIX": (16, "array data type"),
                "BSCALE": (1, ""),
                "BZERO": (32768, ""),
                "EXTEND": (True, ""),
                "HELLO": ("world", ""),  # Specific for first ext
                "NAXIS": (2, "number of array dimensions"),
                "NAXIS1": (2, ""),
                "NAXIS2": (2, ""),
                "SIMPLE": (True, "conforms to FITS standard"),
            },
        ),
        (
            "http://{host}/data/img_multiple.fits",
            "OTHER_IMAGE",
            {
                "BITPIX": (16, "array data type"),
                "BSCALE": (1, ""),
                "BZERO": (32768, ""),
                "EXTNAME": ("OTHER_IMAGE", "extension name"),  # for second ext
                "FOO": ("bar", ""),
                "GCOUNT": (1, "number of groups"),
                "NAXIS": (2, "number of array dimensions"),
                "NAXIS1": (2, ""),
                "NAXIS2": (2, ""),
                "PCOUNT": (0, "number of parameters"),
                "XTENSION": ("IMAGE", "Image extension"),
            },
        ),
    ],
)
def test_load_header(
    with_caching: bool,
    valid_data2d_http_hostname: str,
    filename: str | Path,
    section,
    exp_header,
):
    """Test function 'load_header' with local and remote files."""
    with pyxel.set_options(cache_enabled=with_caching):
        if isinstance(filename, Path):
            # Load header
            header = load_header(filename, section=section)
        else:
            full_url: str = filename.format(host=valid_data2d_http_hostname)

            # Load header
            header = load_header(full_url, section=section)

        # Check
        assert dict(header) == exp_header
