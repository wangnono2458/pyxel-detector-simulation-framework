#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Readout noise model."""

import numpy as np

from pyxel.detectors import APD, Detector
from pyxel.util import set_random_seed


def output_node_noise(
    detector: Detector,
    std_deviation: float,
    seed: int | None = None,
) -> None:
    """Add noise to signal array of detector output node using normal random distribution.

    Parameters
    ----------
    detector : Detector
        Pyxel detector object.
    std_deviation : float
        Standard deviation. Unit: V
    seed : int, optional
        Random seed.

    Raises
    ------
    ValueError
        Raised if 'std_deviation' is negative
    """
    if std_deviation < 0.0:
        raise ValueError("'std_deviation' must be positive.")

    with set_random_seed(seed):
        noise_2d: np.ndarray = np.random.normal(
            scale=std_deviation,
            size=detector.signal.shape,
        )

    detector.signal.array += noise_2d


def compute_readout_noise_saphira(
    roic_readout_noise: float,
    avalanche_gain: float,
    shape: tuple[int, int],
    controller_noise: float = 0.0,
) -> np.ndarray:
    """Compute Saphira specific readout noise.

    Parameters
    ----------
    roic_readout_noise : float
        Readout integrated circuit noise in volts RMS. Unit: V
    avalanche_gain : float
        Avalanche gain.
    shape : tuple
        Shape of the output array.
    controller_noise : float
        Controller noise in volts RMS. Unit: V

    Returns
    -------
    ndarray
    """

    noise_factor = (((1.2 - 1.0) / np.log10(1000)) * np.log10(avalanche_gain)) + 1.0

    total_noise_level = np.sqrt(
        (roic_readout_noise * noise_factor) ** 2 + controller_noise**2
    )

    total_noise = np.random.normal(scale=total_noise_level, size=shape)

    return total_noise


def readout_noise_saphira(
    detector: APD,
    roic_readout_noise: float,
    controller_noise: float = 0.0,
    seed: int | None = None,
) -> None:
    """Apply Saphira specific readout noise to the APD detector.

    Parameters
    ----------
    detector : APD
        Pyxel APD object.
    roic_readout_noise : float
        Readout integrated circuit noise in volts RMS. Unit: V
    controller_noise : float
        Controller noise in volts RMS. Unit: V
    seed : int, optional

    Notes
    -----
    For more information, you can find an example here:
    :external+pyxel_data:doc:`use_cases/APD/saphira`.
    """

    if not isinstance(detector, APD):
        raise TypeError("Expecting a 'APD' detector object.")

    with set_random_seed(seed):
        noise_2d = compute_readout_noise_saphira(
            roic_readout_noise=roic_readout_noise,
            avalanche_gain=detector.characteristics.avalanche_gain,
            shape=detector.geometry.shape,
            controller_noise=controller_noise,
        )

    detector.signal += noise_2d
