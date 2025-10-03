#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


"""Model for loading PSF from file."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from astropy.convolution import convolve_fft

from pyxel.detectors import Detector
from pyxel.inputs import load_image, load_image_v2, load_table_v2

if TYPE_CHECKING:
    import pandas as pd


def apply_psf_2d(
    array_2d: np.ndarray,
    psf_2d: np.ndarray,
    normalize_kernel: bool = True,
) -> np.ndarray:
    """Convolve the input array with the point spread function kernel.

    Parameters
    ----------
    array_2d : ndarray
        Input array to be convolved.
    psf_2d : ndarray
        Convolution kernel representing the point spread function.
    normalize_kernel : bool
        Normalize kernel.

    Returns
    -------
    ndarray
    """
    mean = np.mean(array_2d)

    convolved_array_2d = convolve_fft(
        array_2d,
        kernel=psf_2d,
        boundary="fill",
        fill_value=mean,
        normalize_kernel=normalize_kernel,
        allow_huge=True,
    )

    return convolved_array_2d


def apply_psf_3d(
    array_3d: xr.DataArray,
    psf_2d_or_3d: np.ndarray,
    fill_value: float,
    normalize_kernel: bool = True,
) -> xr.DataArray:
    """Convolve a 3D input array with a 2D or 3D  point spread function (PSF) kernel.

    Parameters
    ----------
    array_3d : DataArray
        Input 3D array to be convolved. The first dimension is typically
        a spectral or time axis, followed by two spatial dimensions.
    psf_2d_or_3d : ndarray
        The point spread function kernel. Must be either:
        - 2D array of shape ``(y, x)``, broadcast to all slices.
        - 3D array of shape ``(n, y, x)``, where ``n`` matches the size of
          the first dimension of `array_3d`.
    fill_value : float
        The constant value to assume outside the array boundaries during convolution.
    normalize_kernel : bool
        If True, the PSF kernel is normalized.

    Returns
    -------
    DataArray
        A new DataArray with the same dimensions, coordinates, and attributes
        as `array_3d`, but with values convolved with the PSF.
    """
    if psf_2d_or_3d.ndim == 2:
        # Convert a 2D PSF into a 3D PSF
        psf_2d: np.ndarray = psf_2d_or_3d

        num_wavelengths, _, _ = array_3d.shape
        psf_size_y, psf_size_x = psf_2d.shape

        new_psf_shape = num_wavelengths, psf_size_y, psf_size_x

        psf_3d: np.ndarray = np.broadcast_to(psf_2d, shape=new_psf_shape)

    elif psf_2d_or_3d.ndim == 3:
        psf_3d = psf_2d_or_3d

        num_wavelengths_psf, _, _ = psf_3d.shape
        num_wavelengths_array, _, _ = array_3d.shape

        if num_wavelengths_psf != num_wavelengths_array:
            raise ValueError(
                f"Mismatch with the number of wavelengths between PSF ({num_wavelengths_psf}) and "
                f"array_3d ({num_wavelengths_array})"
            )

    else:
        raise ValueError(
            f"PSF kernel must be either 2D or 3D, got {psf_2d_or_3d.ndim}D instead."
        )

    convolved_array_3d: np.ndarray = convolve_fft(
        np.asarray(array_3d, dtype=float),
        kernel=psf_3d,
        boundary="fill",
        fill_value=fill_value,
        normalize_kernel=normalize_kernel,
        allow_huge=True,
    )

    convolved_3d = xr.DataArray(
        convolved_array_3d,
        dims=array_3d.dims,
        coords=array_3d.coords,
        attrs=array_3d.attrs,
    )

    return convolved_3d


def load_psf(
    detector: Detector,
    filename: str | Path,
    normalize_kernel: bool = True,
) -> None:
    """Load a point spread function (PSF) from a file and convolve the photon array with the PSF.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    filename : Path or str
        Input filename of the point spread function (PSF).
    normalize_kernel : bool
        If True (default), the PSF kernel is normalized so that its sum is 1.
        If False, the kernel values are used as-is.

    Notes
    -----
    For more information, you can find an example here:
    :external+pyxel_data:doc:`examples/models/scene_generation/tutorial_example_scene_generation`.
    """
    psf_2d: np.ndarray = load_image(filename)

    if detector.photon.ndim == 2:
        detector.photon.array = apply_psf_2d(
            array_2d=detector.photon.array_2d,
            psf_2d=psf_2d,
            normalize_kernel=normalize_kernel,
        )

    elif detector.photon.ndim == 3:
        detector.photon.array_3d = apply_psf_3d(
            array_3d=detector.photon.array_3d,
            psf_2d_or_3d=psf_2d,
            fill_value=float(detector.photon.array_3d.mean()),
            normalize_kernel=normalize_kernel,
        )
    else:
        raise ValueError(
            f"Photon array must be 2D or 3D, got {detector.photon.ndim}D instead."
        )


def load_wavelength_psf(
    detector: Detector,
    filename: str | Path,
    wavelength_col: str,
    y_col: str,
    x_col: str,
    wavelength_table_name: str,
    normalize_kernel: bool = True,
):
    """Read psf files depending on simulation and instrument parameters.

    Parameters
    ----------
    detector : Detector
            Pyxel Detector object.
    filename : Path or str
        Input filename of the point spread function.
    wavelength_col : str
        Dimension name in the file that contains the wavelength information.
    y_col : str
        Dimension name in the file that contains the y information.
    x_col : str
        Dimension name in the file that contains the x information.
    wavelength_table_name : str
        Column name in the file that contains the wavelength information.
    normalize_kernel : bool
            Normalize kernel.
    """
    # load fits image
    kernel_3d: xr.DataArray = load_image_v2(
        filename=filename,
        data_path=0,  # TODO: remove magical value
        rename_dims={"wavelength": wavelength_col, "y": y_col, "x": x_col},
    )

    # load wavelength information from table
    wavelength_table: "pd.DataFrame" = load_table_v2(
        filename=filename,
        data_path=1,  # TODO: remove magical value
        rename_cols={"wavelength": wavelength_table_name},
    )

    # save table information into DataArray.
    wavelength_1d = xr.DataArray(
        wavelength_table["wavelength"],
        dims=["wavelength"],
        coords={"wavelength": wavelength_table["wavelength"]},
        attrs={"units": "nm"},
    )

    # interpolate array along wavelength dimension
    kernel_3d_interpolated: xr.DataArray = (
        kernel_3d.assign_coords(wavelength=wavelength_1d)  # add wavelength(s)
        .interp_like(detector.photon.array_3d)  # interpolate along 'wavelength'
        .dropna(dim="wavelength", how="any")  # drop nan values
    )

    kernel_integrated_2d: xr.DataArray = kernel_3d_interpolated.integrate(
        coord="wavelength"
    )

    # TODO: Check this ! Use 'detector.photon.array_3d' ?
    mean: xr.DataArray = kernel_integrated_2d.mean(dim=["y", "x"])

    # TODO check that kernel size is not to large and kernel has 3 dimensions.
    # if kernel.shape > (200, 50, 50):
    #     raise ValueError("Input PSF used as kernel to convolute with input photon arrat needs to be smaller than "
    #                      "(200, 50, 50). Please reduce the size of the PSF input file, e.g. "
    #                      "with skimage.transform.resize(image, (200, 10, 10)).")

    # convolve the input 3d photon array with the psf kernel
    array_3d: xr.DataArray = apply_psf_3d(
        array_3d=detector.photon.array_3d,
        psf_2d_or_3d=np.asarray(kernel_integrated_2d),
        fill_value=float(mean),
        normalize_kernel=normalize_kernel,
    )

    detector.photon.array_3d = array_3d
