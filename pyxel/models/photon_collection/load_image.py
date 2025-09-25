#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel photon generator models."""

from typing import TYPE_CHECKING, Literal

from pyxel.detectors import Detector
from pyxel.inputs import load_header
from pyxel.util import load_cropped_and_aligned_image

if TYPE_CHECKING:
    from astropy.io import fits


def load_image(
    detector: Detector,
    image_file: str,
    data_path: int | str | None = None,
    include_header: bool = False,
    header_section_index: int | str | None = None,
    position: tuple[int, int] = (0, 0),
    align: (
        Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"] | None
    ) = None,
    convert_to_photons: bool = False,
    multiplier: float = 1.0,
    time_scale: float = 1.0,
    bit_resolution: int | None = None,
) -> None:
    r"""Load an image file (e.g. FITS) as a numpy array and add it to the detector as input Photon array.

    This function reads a 2D image from a variety of file formats (e.g. FITS, netCDF, HDF5, Zarr or ASDF).
    The ``data_path`` parameter is used as an identifier to select the internal data structure within the file-such
    as the HDU (FITS), group (netCDF, HDF5, Zarr) or reference (ASDF).

    The image is cropped, aligned, and optionally converted from ADU to photons using detector-specific characteristics.

    Parameters
    ----------
    detector : Detector
    image_file : str
        Path to image file to load.
    data_path : int or str or None, optional
        Identifier of the dataset within the file. Depending on the file format, this can be:
            * an HDU index or name (for FITS),
            * a group or variable path (for netCDF, HDF5, Zarr),
            * a reference path (for ASDF).

        If ``None``, the default dataset is loaded.
        Ignored for flat (non-hierarchical) formats.
    include_header : bool, optional.
        If ``True``, extract and store header metadata from the image file in the detector object.
        This parameter is provisional and may be removed.
    header_section_index : int or str or None, optional
        Section index or name of the header data to load if `include_header` is enabled.
        This parameter is provisional and may be removed.
    position : tuple of int, optional
        Starting (row, column) indices for placing the image on the detector.
    align : "center", "top_left", "top_right", "bottom_left", "bottom_right"
        Keyword controlling how the image is aligned relative to the detector array.
    convert_to_photons : bool, optional
        If ``True``, converts image values from ADU to photon counts per pixel using the Photon Transfer Function:
        :math:`\mathit{PTF} = \mathit{quantum\_efficiency} \cdot \mathit{charge\_to\_voltage\_conversion}
        \cdot \mathit{pre\_amplification} \cdot \mathit{adc\_factor}`.
    multiplier : float, optional
        Multiplicative scaling factor applied to the photon array.
    time_scale : float, optional
        Time scale of the photon flux, default is 1 second. 0.001 would be ms.
    bit_resolution : int, ootional
        Bit depth of the input image, if `convert_to_photons` is ``True``.

    Notes
    -----
    The image array is cropped and aligned based on the `shape` and `position` parameters and is
    scaled by the `multiplier` and `time_scale` before being added to the detector.

    If `include_header` is enabled, the header metadata is extracted from `image_file` and store in the
    detector`s header storage.
    """
    # TODO: Add units
    shape = (detector.geometry.row, detector.geometry.col)
    position_y, position_x = position

    image = load_cropped_and_aligned_image(
        filename=image_file,
        data_path=data_path,
        shape=shape,
        align=align,
        position_x=position_x,
        position_y=position_y,
    )

    photon_array = image

    if convert_to_photons:
        if not bit_resolution:
            raise ValueError(
                "Bit resolution of the input image has to be specified for converting"
                " to photons."
            )

        cht = detector.characteristics
        adc_multiplier = 2**cht.adc_bit_resolution / 2**bit_resolution

        photon_array = photon_array * adc_multiplier / cht.system_gain

    photon_array = photon_array * (detector.time_step / time_scale) * multiplier

    detector.photon += photon_array

    # Try to extract the Header from 'image_file' as a FITS header even if 'image_file' is not a FITS file
    if include_header:
        header: "fits.Header" | None = load_header(
            image_file, section=header_section_index
        )

        if header:
            if detector.header is None:
                detector.header = header
            else:
                detector.header.update(header)
