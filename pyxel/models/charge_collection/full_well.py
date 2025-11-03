#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel full well models."""

import numpy as np

from pyxel.detectors import Detector


def apply_simple_full_well_capacity(array: np.ndarray, fwc: int) -> np.ndarray:
    """Apply a hard limit corresponding to the full well capacity (FWC) to an array.

    This function enforces a saturation limit on each element of the input array.
    Any values greater than the specified full well capacity are clipped to ``fwc``.
    Negative or zero values are left unchanged.

    Parameters
    ----------
    array : ndarray
        Input array representing pixel charge values
    fwc:  int
        Full well capacity value defining the maximum allowed charge per pixel.

    Returns
    -------
    ndarray
        New array with charge values limited to the full well capacity.
    """
    array_with_fwc = np.where(array > fwc, fwc, array)
    return array_with_fwc


def simple_full_well(detector: Detector, fwc: int | None = None) -> None:
    """Apply the full well capacity limit to detector `pixel` bucket.

    Uses full well capacity in the characteristics of the detector object if not overridden by the function argument.

    Parameters
    ----------
    detector : Detector
    fwc : int, optional
        Full well capacity (in electrons).
        If not provided, the value is read from ``detector.characteristics.full_well_capacity``.

    Notes
    -----
    This model assumes an ideal clipping behavior, where excess charge is discarded
    once the pixel reaches saturation. More complex models may redistribute or
    model blooming effects across neighboring pixels.
    """
    # Retrieve the full well capacity value either from argument or detector characteristics.
    if fwc is None:
        fwc_input = detector.characteristics.full_well_capacity
    else:
        fwc_input = fwc

    # Check for invalid (negative) full well capacity input.
    if fwc_input < 0:
        raise ValueError("Full well capacity should be a positive number.")

    # Apply clipping according to full well capacity.
    charge_2d = apply_simple_full_well_capacity(
        array=detector.pixel.non_volatile.array,
        fwc=fwc_input,
    )

    detector.pixel.non_volatile.array = charge_2d
