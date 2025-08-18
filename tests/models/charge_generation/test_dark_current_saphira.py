#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Tests for dark current models."""

import pytest

from pyxel.detectors import (
    APD,
    CCD,
    APDCharacteristics,
    APDGeometry,
    CCDGeometry,
    Characteristics,
    Environment,
    ReadoutProperties,
)
from pyxel.detectors.apd import AvalancheSettings, ConverterFunction, ConverterValues
from pyxel.models.charge_generation import dark_current_saphira


@pytest.fixture
def ccd_10x10() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=10,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(temperature=200.0),
        characteristics=Characteristics(),
    )
    detector._readout_properties = ReadoutProperties(times=[1.0])
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
        environment=Environment(temperature=50.0),
        characteristics=APDCharacteristics(
            roic_gain=0.8,
            bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
            avalanche_settings=AvalancheSettings(
                avalanche_gain=10.0,
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
    detector._readout_properties = ReadoutProperties(times=[1.0])
    return detector


def test_dark_current_saphira_valid(apd_5x5: APD):
    """Test model 'dark_current_saphira' with valid inputs."""
    dark_current_saphira(detector=apd_5x5)


def test_dark_current_saphira_with_ccd(ccd_10x10: CCD):
    """Test model 'dark_current_saphira' with a 'CCD'."""
    detector = ccd_10x10

    with pytest.raises(TypeError, match="Expecting an APD object for detector."):
        dark_current_saphira(detector=detector)


@pytest.mark.parametrize(
    "temperature, gain, exp_exc, exp_error",
    [
        pytest.param(
            10.0,
            1.0,
            ValueError,
            "Dark current is inaccurate for avalanche gains less than 2!",
        ),
        pytest.param(
            200.0,
            10.0,
            ValueError,
            "Dark current estimation is inaccurate for temperatures more than 100 K!",
        ),
    ],
)
def test_dark_current_saphira_invalid(
    apd_5x5: APD, temperature: float, gain: float, exp_exc, exp_error
):
    """Test model 'dark_current_saphira' with valid inputs."""
    detector = apd_5x5
    with pytest.raises(exp_exc, match=exp_error):  # noqa
        detector.environment.temperature = temperature
        detector.characteristics.avalanche_gain = gain
        dark_current_saphira(detector=detector)
