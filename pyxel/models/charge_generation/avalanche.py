# Copyright or © or Copr. Jean Le Graët, CEA Paris-Saclay (2025)
#
# jean.legraet@cea.fr
#
# This file is part of the Pyxel general simulator framework.
#
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

"""
APD Avalanche gain utility functions.

@author: jlegraet
"""


from collections.abc import Callable

import numpy as np

from pyxel.detectors import APD
from pyxel.util import set_random_seed


def calculate_avalanche_gain(
    charge_array_2d: np.ndarray,
    init_apd_bias: float,  # PRV - common ==> avalanche_bias
    charge_to_volt_conversion: float,
    bias_to_gain: Callable,
) -> np.ndarray:
    r"""Calculate avalanche gain map from charge array and initial avalanche for an APD detector.

    The avalanche gain is derived for the bias voltage across the APD junction.
    The bias decreases as charge accumulated following:

    :math:`V_{apd} = V_{init} - Q . \frac{dV}{dQ}`

    The resulting voltage is then converted to a gain

    Parameters
    ----------
    charge_array_2d : np.ndarray
        2D charge array of accumulated charges (in electron) for each pixel.
    init_apd_bias : float
        Initial avalanche bias voltage applied in the APD (in V).
    charge_to_volt_conversion : float
        Charge-to-volt conversion factor (in V/electron)
    bias_to_gain : callable
        Function that converts the APD bias (in V) to a gain value

    Returns
    -------
    np.ndarray
        2D Avalanche gain array.
    """
    apd_bias_2d = init_apd_bias - charge_array_2d * charge_to_volt_conversion
    avalanche_gain_2d = bias_to_gain(apd_bias_2d)

    return avalanche_gain_2d


def avalanche_gain(
    detector: APD,
    excess_noise_factor: float = 1.0,
    seed: int | None = None,
) -> None:
    """Apply Avalanche multiplication gain to the charge array of the detector.

    This model simulates the avalanche amplification process occurring in Avalanche Photodiodes (APDs).
    An excess noise factor can be included for statistical variations in the avalanche process.

    Parameters
    ----------
    detector : Detector
        Pyxel Detector object.
    excess_noise_factor : float, optional. Default is 1.0
        Multiplication excess noise factor.
        When equal to 1.0, the avalanche gain is deterministic.
        When > 1.0, gain fluctuations are modeled using a Gamma distribution.
    seed : int, optional
        Random seed used when simulating stochastic gain variations.
    """
    mean_avalanche_gain_2d = calculate_avalanche_gain(
        charge_array_2d=np.array(detector.pixel),
        init_apd_bias=detector.characteristics.avalanche_settings.avalanche_bias,
        charge_to_volt_conversion=detector.characteristics.charge_to_volt_conversion,
        bias_to_gain=detector.characteristics.avalanche_settings.bias_to_gain,
    )

    array_copy = detector.charge.array.copy()

    if excess_noise_factor == 1.0:
        out = mean_avalanche_gain_2d * array_copy
    else:
        r = 1.0 / (excess_noise_factor - 1.0)
        gamma_shape = r * array_copy
        gamma_scale = mean_avalanche_gain_2d / r

        with set_random_seed(seed):
            out = np.random.gamma(shape=gamma_shape, scale=gamma_scale)

    detector.charge.empty()
    detector.charge.add_charge_array(array=out)
