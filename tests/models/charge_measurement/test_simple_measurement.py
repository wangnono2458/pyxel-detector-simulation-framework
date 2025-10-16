#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import numpy as np
import pytest

from pyxel.detectors import CCD, CCDGeometry, Characteristics, Environment
from pyxel.detectors.channels import Channels, Matrix, ReadoutPosition
from pyxel.models.charge_measurement import simple_measurement


@pytest.fixture
def ccd_5x10() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=2,
            col=3,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )
    detector.signal.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


@pytest.mark.parametrize("gain", [None, 0.8])
def test_simple_measurement(ccd_5x10: CCD, gain):
    """Test model 'simple_measurement."""
    pixel_2d = np.array(
        [[0.22733602, 0.31675834, 0.79736546], [0.67625467, 0.39110955, 0.33281393]],
    )

    detector = ccd_5x10
    detector.pixel.array = pixel_2d.copy()
    detector.characteristics.charge_to_volt_conversion = 0.5

    if gain is None:
        simple_measurement(detector)
        exp_signal = np.array(
            [
                [0.11366801, 0.15837917, 0.39868273],
                [0.33812734, 0.19555478, 0.16640696],
            ],
        )
    else:
        simple_measurement(detector, gain=gain)
        exp_signal = np.array(
            [
                [0.18186882, 0.25340667, 0.63789237],
                [0.54100374, 0.31288764, 0.26625114],
            ],
        )

    signal = detector.signal.array
    np.testing.assert_allclose(actual=signal, desired=exp_signal, rtol=1e-6)


@pytest.fixture
def ccd_2x2_with_channels() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=2,
            col=2,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
            channels=Channels(
                matrix=Matrix([["OP9", "OP13"], ["OP1", "OP5"]]),
                readout_position=ReadoutPosition(
                    {
                        "OP9": "top-left",
                        "OP13": "top-left",
                        "OP1": "bottom-left",
                        "OP5": "bottom-left",
                    }
                ),
            ),
        ),
        environment=Environment(),
        characteristics=Characteristics(
            charge_to_volt_conversion=1e-6,
            pre_amplification={"OP9": 1.0, "OP13": 2.0, "OP1": 3.0, "OP5": 4.0},
        ),
    )
    detector.signal.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


@pytest.mark.parametrize("gain", [None, 0.8])
def test_simple_measurement_with_channels(ccd_2x2_with_channels: CCD, gain):
    """Test model 'simple_measurement."""
    pixel_2d = np.array(
        [[0.22733602, 0.31675834], [0.67625467, 0.39110955]],
    )

    detector = ccd_2x2_with_channels
    detector.pixel.array = pixel_2d.copy()
    detector.characteristics.charge_to_volt_conversion = (
        detector.characteristics.charge_to_volt_conversion
    )

    if gain is None:
        simple_measurement(detector)
        exp_signal = np.array(
            [[2.2733602e-07, 3.1675834e-07], [6.7625467e-07, 3.9110955e-07]]
        )
    else:
        simple_measurement(detector, gain=gain)
        exp_signal = np.array(
            [
                [0.18186882, 0.25340667],
                [0.54100374, 0.31288764],
            ],
        )

    signal = detector.signal.array
    np.testing.assert_allclose(actual=signal, desired=exp_signal, rtol=1e-6)
