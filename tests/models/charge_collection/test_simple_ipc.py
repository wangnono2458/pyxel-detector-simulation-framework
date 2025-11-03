#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


"""Tests for simple inter-pixel capacitance model."""

import numpy as np
import pytest

from pyxel.detectors import (
    CCD,
    CMOS,
    CCDGeometry,
    Characteristics,
    CMOSGeometry,
    Environment,
)
from pyxel.models.charge_collection import simple_ipc
from pyxel.models.charge_collection.inter_pixel_capacitance import (
    compute_ipc_convolution,
    ipc_kernel,
)


@pytest.fixture
def cmos_5x5() -> CMOS:
    """Create a valid CCD detector."""
    detector = CMOS(
        geometry=CMOSGeometry(row=5, col=5),
        environment=Environment(),
        characteristics=Characteristics(),
    )
    detector.pixel.non_volatile.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


def test_ipc_kernel():
    """Test function 'ipc_kernel'."""
    result_2d = ipc_kernel(
        coupling=0.1,
        diagonal_coupling=0.05,
        anisotropic_coupling=0.03,
    )

    exp_2d = np.array([[0.05, 0.07, 0.05], [0.13, 0.4, 0.13], [0.05, 0.07, 0.05]])
    np.testing.assert_allclose(result_2d, exp_2d)


def test_compute_ipc_convolution():
    """Test function 'compute_ipc_convolution'."""
    data_2d = np.array(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 00.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected_array = np.array(
        [
            [-4.46, 2.048, 0.748, 0.748, 1.54],
            [1.712, 0.5, 5.00, 7.00, 6.012],
            [1.012, 0.0, 13.0, -60.0, 14.012],
            [1.012, 0.0, 5.00, 7.00, 6.012],
            [1.54, 0.748, 0.748, 0.748, 1.54],
        ]
    )

    result_2d = compute_ipc_convolution(
        data_2d, coupling=0.1, diagonal_coupling=0.05, anisotropic_coupling=0.03
    )

    np.testing.assert_allclose(result_2d, expected_array, atol=1.0)


def test_simple_ipc_valid(cmos_5x5: CMOS):
    """Test model 'simple_ipc' with valid inputs."""
    detector = cmos_5x5
    data_2d = np.array(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 00.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    detector.pixel.non_volatile.array = np.array(data_2d)

    expected_array = np.array(
        [
            [-4.46, 2.048, 0.748, 0.748, 1.54],
            [1.712, 0.5, 5.00, 7.00, 6.012],
            [1.012, 0.0, 13.0, -60.0, 14.012],
            [1.012, 0.0, 5.00, 7.00, 6.012],
            [1.54, 0.748, 0.748, 0.748, 1.54],
        ]
    )

    # Check before applying the model
    with pytest.raises(ValueError, match=r"not initialized"):
        _ = detector.pixel.volatile.array

    # Apply model
    simple_ipc(
        detector=cmos_5x5,
        coupling=0.1,
        diagonal_coupling=0.05,
        anisotropic_coupling=0.03,
    )

    np.testing.assert_allclose(detector.pixel.non_volatile, data_2d)
    np.testing.assert_allclose(detector.pixel.volatile, expected_array, atol=1.0)


@pytest.mark.parametrize(
    "coupling, diagonal_coupling, anisotropic_coupling, exp_exc, exp_error",
    [
        pytest.param(
            0.05,
            0.08,
            0.01,
            ValueError,
            "Requirement diagonal_coupling <= coupling is not met.",
        ),
        pytest.param(
            0.03,
            0.02,
            0.05,
            ValueError,
            "Requirement anisotropic_coupling <= coupling is not met.",
        ),
        pytest.param(
            0.2,
            0.1,
            0.01,
            ValueError,
            r"Requirement coupling \+ diagonal_coupling << 1 is not met.",
        ),
    ],
)
def test_charge_blocks_inputs(
    cmos_5x5: CMOS,
    coupling: float,
    diagonal_coupling: float,
    anisotropic_coupling: float,
    exp_exc,
    exp_error,
):
    """Test model 'charge_blocks' with bad inputs."""
    with pytest.raises(exp_exc, match=exp_error):
        simple_ipc(
            detector=cmos_5x5,
            coupling=coupling,
            diagonal_coupling=diagonal_coupling,
            anisotropic_coupling=anisotropic_coupling,
        )


def test_charge_blocks_with_ccd():
    """Test model 'charge_blocks' with a `CCD` detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=10,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )

    with pytest.raises(TypeError, match="Expecting a CMOS object for detector."):
        simple_ipc(
            detector=detector,
            coupling=0.1,
            diagonal_coupling=0.05,
            anisotropic_coupling=0.03,
        )
