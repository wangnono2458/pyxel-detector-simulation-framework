#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Charge readout model."""

from astropy.units import Quantity

from pyxel.detectors import Detector


def apply_gain(pixel_2d: Quantity, gain: Quantity) -> Quantity:
    """Apply an electronic gain (in V/e-) to convert PixelRead charges (in e-) into Signal (in V).

    Parameters
    ----------
    pixel_2d : ndarray
        2D array of pixels. Unit: e-
    gain : float
        Gain factor to apply. Unit: V/e-

    Returns
    -------
    Quantity
        2D array of signals. Unit: V
    """
    new_data_2d = pixel_2d * gain
    return new_data_2d.to("V")


def simple_measurement(detector: Detector, gain: float | None = None) -> None:
    """Convert detector PixelRead charge values (in electron) into Signal values (in Volt) using a specified gain.

    Notes
    -----
    If no gain is provided, the detector's internal ``detector.characteristics.charge_to_volt_conversion`` parameter is used.

    Parameters
    ----------
    detector : Detector
        PyxelRead Detector object.
    gain : float, optional
        Gain factor to apply. If not provided, the default is ``detector.characteristics.charge_to_volt_conversion``. Unit: V/e-
    """

    if gain is None:
        gain_to_apply = Quantity(
            detector.characteristics.charge_to_volt_conversion,
            unit="V/electron",
        )

    else:
        gain_to_apply = Quantity(gain, unit="V/electron")

    detector.signal = apply_gain(pixel_2d=Quantity(detector.pixel), gain=gain_to_apply)
