#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Tests for full well models."""

import numpy as np
import pytest

from pyxel.detectors import CCD, CCDGeometry, Characteristics, Environment
from pyxel.models.charge_collection import simple_full_well


@pytest.fixture
def ccd_2x3() -> CCD:
    """Create a valid CCD detector."""
    return CCD(
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


@pytest.mark.parametrize("fwc", [10, None])
def test_full_well(ccd_2x3: CCD, fwc: int):
    """Test model 'simple_full_well' with valid inputs."""

    detector = ccd_2x3
    detector.characteristics.full_well_capacity = 10

    detector.pixel.non_volatile.array = np.array(
        [[5, 7, 9.9], [10.1, 11, 20]],
        dtype=float,
    )

    # Check before applying the model
    with pytest.raises(ValueError, match=r"not initialized"):
        _ = detector.pixel.volatile.array

    # Apply model
    simple_full_well(detector=ccd_2x3, fwc=fwc)

    # Check outputs
    exp_non_volatile = np.array([[5, 7, 9.9], [10, 10, 10]], dtype=float)

    with pytest.raises(ValueError, match=r"not initialized"):
        _ = detector.pixel.volatile.array

    np.testing.assert_equal(detector.pixel.non_volatile, exp_non_volatile)
    np.testing.assert_equal(detector.pixel, exp_non_volatile)


@pytest.mark.parametrize(
    "fwc, exp_exc, exp_error",
    [
        pytest.param(
            -5,
            ValueError,
            "Full well capacity should be a positive number.",
        ),
    ],
)
def test_full_well_bad_inputs(
    ccd_2x3: CCD,
    fwc: int,
    exp_exc,
    exp_error,
):
    """Test model 'simple_full_well' with bad inputs."""
    with pytest.raises(exp_exc, match=exp_error):
        simple_full_well(detector=ccd_2x3, fwc=fwc)
