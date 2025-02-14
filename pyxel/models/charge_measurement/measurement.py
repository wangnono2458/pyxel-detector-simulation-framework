#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Charge readout model."""

import numpy as np

from pyxel.detectors import Detector


def apply_gain(pixel_2d: np.ndarray, gain: float) -> np.ndarray:
    """Apply a gain (in V/e-) to a pixel array (in e-).

    Parameters
    ----------
    pixel_2d : ndarray
        2D array of pixels. Unit: e-
    gain : float
        Gain to apply. Unit: V/e-

    Returns
    -------
    ndarray
        2D array of signals. Unit: V
    """
    new_data_2d = pixel_2d * gain

    return new_data_2d


def simple_measurement(detector: Detector, gain: float | None = None) -> None:
    """Convert the pixel array into signal array.

    Notes
    -----
    If no gain is provided, then its value will be the sensitivity of charge readout
    provided in the ``Detector`` object.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    gain : float, optional
        Gain to apply. By default, this is the sensitivity of charge readout. Unit: V/e-
    """
    if gain is None:
        gain = detector.characteristics._channels_gain

    # Apply a gain (in V/e-) to a pixel array (in e-)
    detector.signal.array = np.asarray(detector.pixel.array * gain, dtype=float)
