#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""SAPHIRA HgCdTe e-APD avalanche photodiode (APD) characteristic models.

References
----------
[1] Leonardo MW Ltd., Electrical Interface Document for SAPHIRA (ME1000) in a 68-pin LLC
(3313-520-), issue 1A, 2012.

[2] I. Pastrana et al., HgCdTe SAPHIRA arrays: individual pixel measurement of charge gain and
node capacitance utilizing a stable IR LED, in High Energy, Optical, and Infrared Detectors for
Astronomy VIII, 2018, vol. 10709, no. July 2018, 2018, p. 37.

[3] S. B. Goebel et al., Overview of the SAPHIRA detector for adaptive optics applications, in
Journal of Astronomical Telescopes, Instruments, and Systems, 2018, vol. 4, no. 02, p. 1.

[4] G. Finger et al., Sub-electron read noise and millisecond full-frame readout with the near
infrared eAPD array SAPHIRA, in Adaptive Optics Systems V, 2016, vol. 9909, no. July 2016, p.
990912.

[5] I. M. Baker et al., Linear-mode avalanche photodiode arrays in HgCdTe at Leonardo, UK: the
current status, in Image Sensing Technologies: Materials, Devices, Systems, and Applications VI,
2019, vol. 10980, no. May, p. 20.
"""

import numpy as np

from pyxel.detectors import APDCharacteristics
from pyxel.detectors.apd import AvalancheSettings, ConverterFunction


def bias_to_node_capacitance_saphira(bias: float) -> float:
    """Compute the pixel integrating node capacitance of a SAPHIRA detector.

    The below interpolates empirical published data, however note that
    Node C = Charge Gain / Voltage Gain
    So can be calculated by measuring V gain (varying PRV) and chg gain (PTC); see [2]

    Parameters
    ----------
    bias : float
        Detector bias voltage in V

    Returns
    -------
    float
        Capacitance in F
    """
    # Late import
    from astropy.units import Quantity

    if bias < 1:
        raise ValueError(
            "Warning! Node capacitance calculation is inaccurate for bias voltages"
            " <1 V!"
        )

    # From [2] (Mk13 ME1000; data supplied by Leonardo):
    bias_voltages = Quantity(
        [1.0, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 10.5],
        unit="V",
    )
    node_capacitances = Quantity(
        [46.5, 41.3, 37.3, 34.8, 33.2, 31.4, 30.7, 30.4],
        unit="fF",
    )

    # Get an 'output_capacitance' in 'fF'
    output_capacitance: Quantity = np.interp(
        x=Quantity(bias, unit="V"),
        xp=bias_voltages,
        fp=node_capacitances,
    )

    # Convert it into 'F'
    return output_capacitance.to("F").value


def gain_to_bias_saphira(gain: float) -> float:
    """Convert avalanche gain to detector bias for a SAPHIRA detector.

    The formula ignores the soft knee between the linear and
    unity gain ranges, but should be close enough. [2] (Mk13 ME1000)

    Parameters
    ----------
    gain : float
        Avalanche gain.

    Returns
    -------
    float
        Estimated bias voltage in V.
    """
    # Late import
    import math

    bias = (2.17 * math.log2(gain)) + 2.65

    return bias


def bias_to_gain_saphira(bias: float) -> float:
    """Convert detector bias to avalanche gain for a SAPHIRA detector.

    The formula ignores the soft knee between the linear and unity gain ranges,
    but should be close enough. [2] (Mk13 ME1000)

    Parameters
    ----------
    bias : float
        Bias voltage in V.

    Returns
    -------
    float
        Avalanche gain.
    """
    gain = 2 ** ((bias - 2.65) / 2.17)

    if gain < 1.0:
        gain = 1.0  # Unity gain is lowest

    return gain


class SaphiraCharacteristics(APDCharacteristics):
    """SAPHIRA HgCdTe e-APD avalanche photodiode (APD) detector characteristics.

    Parameters
    ----------
    roic_gain : float
        Gain of the read-out integrated circuit. Unit: V/V
    avalanche_gain : float, optional
        APD gain. Unit: electron/electron
    pixel_reset_voltage : float, optional
        DC voltage going into the detector, not the voltage of a reset pixel. Unit: V
    common_voltage : float, optional
        Common voltage. Unit: V
    quantum_efficiency : float, optional
        Quantum efficiency.
    full_well_capacity : float, optional
        Full well capacity. Unit: e-
    adc_bit_resolution : int, optional
        ADC bit resolution.
    adc_voltage_range : tuple of float, optional
        ADC voltage range. Unit: V
    """

    def __init__(
        self,
        roic_gain: float,  # unit: V
        avalanche_gain: float | None = None,
        pixel_reset_voltage: float | None = None,
        common_voltage: float | None = None,
        #####################
        # Common parameters #
        #####################
        quantum_efficiency: float | None = None,  # unit: NA
        full_well_capacity: float | None = None,  # unit: electron
        adc_bit_resolution: int | None = None,
        adc_voltage_range: tuple[float, float] | None = None,  # unit: V
    ):
        super().__init__(
            roic_gain=roic_gain,
            bias_to_node=ConverterFunction(bias_to_node_capacitance_saphira),
            avalanche_settings=AvalancheSettings(
                gain_to_bias=ConverterFunction(gain_to_bias_saphira),
                bias_to_gain=ConverterFunction(bias_to_gain_saphira),
                avalanche_gain=avalanche_gain,
                pixel_reset_voltage=pixel_reset_voltage,
                common_voltage=common_voltage,
            ),
            quantum_efficiency=quantum_efficiency,
            full_well_capacity=full_well_capacity,
            adc_bit_resolution=adc_bit_resolution,
            adc_voltage_range=adc_voltage_range,
        )
