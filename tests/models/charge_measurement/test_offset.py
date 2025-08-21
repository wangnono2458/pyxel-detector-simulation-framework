#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


"""Voltage offset model tests."""

import numpy as np
import pytest

from pyxel.detectors import (
    APD,
    CCD,
    APDCharacteristics,
    APDGeometry,
    CCDGeometry,
    Characteristics,
    Environment,
)
from pyxel.detectors.apd import AvalancheSettings, ConverterFunction, ConverterValues
from pyxel.models.charge_measurement import dc_offset, output_pixel_reset_voltage_apd


@pytest.fixture
def ccd_5x5() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=5,
            col=5,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(
            adc_voltage_range=(0.0, 10.0),
        ),
    )
    detector.signal.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


@pytest.fixture
def apd_5x5() -> APD:
    """Create a valid CCD detector."""
    detector = APD(
        geometry=APDGeometry(
            row=5,
            col=5,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=APDCharacteristics(
            roic_gain=0.8,
            bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
            avalanche_settings=AvalancheSettings(
                avalanche_gain=1.0,
                pixel_reset_voltage=5.0,
                gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
            ),
            quantum_efficiency=1.0,
            adc_voltage_range=(0.0, 10.0),
            adc_bit_resolution=16,
            full_well_capacity=100000,
        ),
    )
    detector.signal.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


@pytest.mark.parametrize("detector_type", ["ccd", "apd"])
@pytest.mark.parametrize(
    "voltage",
    [pytest.param(0.1, id="ccd"), pytest.param(3.0, id="apd")],
)
def test_offset(
    ccd_5x5: CCD,
    apd_5x5: APD,
    detector_type: str,
    voltage: float,
):
    """Test offset models 'dc_offset' and 'output_pixel_reset_voltage_apd' with valid inputs."""

    if detector_type == "ccd":
        dc_offset(detector=ccd_5x5, offset=voltage)
    elif detector_type == "apd":
        output_pixel_reset_voltage_apd(detector=apd_5x5, roic_drop=voltage)
    else:
        raise NotImplementedError


@pytest.mark.parametrize(
    "voltage, exp_exc, exp_error",
    [
        pytest.param(
            20.0,
            ValueError,
            "Parameter offset=(.*) V out of bonds of the ADC voltage range.",
        ),
        pytest.param(
            -1.0,
            ValueError,
            "Parameter offset=(.*) V out of bonds of the ADC voltage range.",
        ),
    ],
)
def test_dc_offset_invalid(ccd_5x5: CCD, voltage: float, exp_exc, exp_error):
    """Test model 'dc_offset' with valid inputs."""
    detector = ccd_5x5
    with pytest.raises(exp_exc, match=exp_error):
        dc_offset(detector=detector, offset=voltage)


@pytest.mark.parametrize(
    "drop, exp_exc, exp_error",
    [
        pytest.param(
            10.0,
            ValueError,
            "Output pixel reset voltage out of bonds of the ADC voltage range.",
        ),
    ],
)
def test_output_pixel_reset_voltage_apd_invalid(
    apd_5x5: APD, drop: float, exp_exc, exp_error
):
    """Test model 'output_pixel_reset_voltage_apd' with valid inputs."""
    detector = apd_5x5
    with pytest.raises(exp_exc, match=exp_error):
        output_pixel_reset_voltage_apd(detector=detector, roic_drop=drop)


def test_output_pixel_reset_voltage_apd_invalid_with_ccd(ccd_5x5: CCD):
    """Test model 'output_pixel_reset_voltage_apd' with a 'CCD'."""
    detector = ccd_5x5

    with pytest.raises(TypeError, match="Expecting a 'APD' detector object."):
        output_pixel_reset_voltage_apd(detector=detector, roic_drop=1.0)
