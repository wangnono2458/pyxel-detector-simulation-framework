#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


"""Reset noise model tests."""

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
from pyxel.models.charge_measurement import ktc_noise


@pytest.fixture
def ccd_2x3() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=2,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(temperature=273.0),
        characteristics=Characteristics(
            adc_voltage_range=(0.0, 10.0),
        ),
    )
    detector.signal.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


@pytest.fixture
def apd_2x3() -> APD:
    """Create a valid CCD detector."""
    detector = APD(
        geometry=APDGeometry(
            row=2,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(temperature=273.0),
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


@pytest.mark.parametrize("detector_name", ["ccd", "apd"])
@pytest.mark.parametrize(
    "non_destructive",
    [
        pytest.param(False, id="destructive"),
        pytest.param(True, id="non-destructive"),
    ],
)
def test_ktc_noise(
    detector_name: str,
    non_destructive: bool,
    ccd_2x3: CCD,
    apd_2x3: APD,
):
    """Test model 'ktc_noise' with valid inputs."""

    if detector_name == "ccd":
        detector = ccd_2x3
    elif detector_name == "apd":
        detector = apd_2x3
    else:
        raise ValueError

    detector.set_readout(
        times=[1.1, 2.2, 3.3, 4.4],
        start_time=0.5,
        non_destructive=non_destructive,
    )
    detector.empty()
    detector.signal.array = np.zeros(shape=(2, 3), dtype=float)

    #
    # First exposure
    #
    detector.readout_properties.pipeline_count = 0
    ktc_noise(detector=detector, node_capacitance=30.0e-15, seed=12345)

    actual_signal_2d = detector.signal.array
    exp_signal_2d = np.array(
        [
            [-7.25598590e-05, 1.69764342e-04, -1.84118171e-04],
            [-1.96981943e-04, 6.96782727e-04, 4.93901064e-04],
        ]
    )
    np.testing.assert_allclose(
        actual=actual_signal_2d, desired=exp_signal_2d, rtol=1e-5
    )

    #
    # Second exposure
    #
    detector.readout_properties.pipeline_count = 1
    ktc_noise(detector=detector, node_capacitance=30.0e-15, seed=12345)

    actual_signal_2d = detector.signal.array
    if non_destructive:
        # exp_signal_2d is unchanged
        pass
    else:
        # Destructive readout
        exp_signal_2d = np.array(
            [
                [-0.00014512, 0.00033953, -0.00036824],
                [-0.00039396, 0.00139357, 0.0009878],
            ]
        )

    np.testing.assert_allclose(
        actual=actual_signal_2d, desired=exp_signal_2d, rtol=1e-5
    )

    #
    # Third exposure
    #
    detector.readout_properties.pipeline_count = 2
    ktc_noise(detector=detector, node_capacitance=30.0e-15, seed=12345)

    actual_signal_2d = detector.signal.array
    if non_destructive:
        # exp_signal_2d is unchanged
        pass
    else:
        # Destructive readout
        exp_signal_2d = np.array(
            [
                [-0.00021768, 0.00050929, -0.00055235],
                [-0.00059095, 0.00209035, 0.0014817],
            ]
        )

    np.testing.assert_allclose(
        actual=actual_signal_2d, desired=exp_signal_2d, rtol=1e-5
    )


@pytest.mark.parametrize(
    "node_capacitance, exp_exc, exp_error",
    [
        pytest.param(
            -30e-15,
            ValueError,
            "Node capacitance should be larger than 0!",
        ),
        pytest.param(
            None,
            AttributeError,
            "Characteristic node_capacitance not available for the detector used. "
            "Please specify node_capacitance in the model argument!",
        ),
    ],
)
def test_ktc_noise_invalid(
    ccd_2x3: CCD,
    node_capacitance: float | None,
    exp_exc,
    exp_error,
):
    """Test model 'output_pixel_reset_voltage_apd' with valid inputs."""
    detector = ccd_2x3
    with pytest.raises(exp_exc, match=exp_error):
        ktc_noise(detector=detector, node_capacitance=node_capacitance)
