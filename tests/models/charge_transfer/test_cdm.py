#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytest
from astropy import constants

from pyxel.detectors import (
    CCD,
    CMOS,
    CCDGeometry,
    Characteristics,
    CMOSGeometry,
    Environment,
)
from pyxel.models.charge_transfer import cdm
from pyxel.models.charge_transfer.cdm import run_cdm_parallel


@pytest.fixture
def ccd_5x5() -> CCD:
    """Create a valid CCD detector."""
    return CCD(
        geometry=CCDGeometry(
            row=5,
            col=5,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(temperature=273.15),
        characteristics=Characteristics(full_well_capacity=100_000),
    )


@pytest.fixture
def input_array() -> np.ndarray:
    """Create a valid input_array."""
    out = np.ones((5, 5), dtype=float) * 1000
    out[:3, :] *= 0
    return out


@pytest.mark.parametrize(
    """
    direction,
    trap_release_times,
    trap_densities,
    sigma,
    beta,
    full_well_capacity,
    max_electron_volume,
    transfer_period,
    charge_injection,
    exp_err,
    exp_exc
    """,
    [
        pytest.param(
            "parallel",
            [1.0],
            [1.0],
            [1.0],
            1.0,
            1.0,
            2.0,
            1.0,
            False,
            ValueError,
            r"'max_electron_volume' must be between 0.0 and 1.0.",
            id="Max volume out of bonds.",
        ),
        pytest.param(
            "parallel",
            [1.0],
            [1.0],
            [1.0],
            2.0,
            1.0,
            1.0,
            1.0,
            False,
            ValueError,
            r"'beta' must be between 0.0 and 1.0.",
            id="Beta out of bonds.",
        ),
        pytest.param(
            "parallel",
            [1.0],
            [1.0],
            [1.0],
            1.0,
            1.0e8,
            1.0,
            1.0,
            False,
            ValueError,
            "'full_well_capacity' must be between 0 and 1e7.",
            id="Fwc out of bonds.",
        ),
        pytest.param(
            "parallel",
            [1.0],
            [1.0],
            [1.0],
            1.0,
            1.0,
            1.0,
            20.0,
            False,
            ValueError,
            r"'transfer_period' must be between 0.0 and 10.0.",
            id="Transfer period out of bonds.",
        ),
        pytest.param(
            "parallel",
            [1.0, 2.0],
            [1.0],
            [1.0],
            1.0,
            1.0,
            1.0,
            1.0,
            False,
            ValueError,
            r"Length of 'sigma', 'trap_densities' and 'trap_release_times' not the"
            r" same!",
            id="Different lengths.",
        ),
        pytest.param(
            "parallel",
            [],
            [],
            [],
            1.0,
            1.0,
            1.0,
            1.0,
            False,
            ValueError,
            r"Expecting inputs for at least one trap species.",
            id="Empty.",
        ),
    ],
)
def test_cdm_bad_inputs(
    ccd_5x5: CCD,
    direction: Literal["parallel", "serial"],
    beta: float,
    trap_release_times: Sequence[float],
    trap_densities: Sequence[float],
    sigma: Sequence[float],
    full_well_capacity: float | None,
    max_electron_volume: float,
    transfer_period: float,
    charge_injection: bool,
    exp_err,
    exp_exc,
):
    """Test function 'cdm' with bad inputs."""
    with pytest.raises(expected_exception=exp_err, match=exp_exc):
        cdm(
            detector=ccd_5x5,
            direction=direction,
            trap_release_times=trap_release_times,
            trap_densities=trap_densities,
            sigma=sigma,
            beta=beta,
            full_well_capacity=full_well_capacity,
            max_electron_volume=max_electron_volume,
            transfer_period=transfer_period,
            charge_injection=charge_injection,
        )


def test_cdm_with_cmos():
    """Test function 'cdm' with CMOS."""

    detector = CMOS(
        geometry=CMOSGeometry(
            row=10,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )

    with pytest.raises(
        expected_exception=TypeError, match="Expecting a `CCD` object for 'detector'."
    ):
        cdm(
            detector=detector,
            direction="parallel",
            trap_release_times=[1.0],
            trap_densities=[1.0],
            sigma=[1.0],
            beta=1.0,
            full_well_capacity=1.0,
            max_electron_volume=1.0,
            transfer_period=1.0,
            charge_injection=False,
        )


@pytest.mark.parametrize(
    """
    direction,
    trap_release_times,
    trap_densities,
    sigma,
    beta,
    full_well_capacity,
    max_electron_volume,
    transfer_period,
    charge_injection,
    """,
    [
        pytest.param(
            "parallel",
            [5.0e-3, 5.0e-3],
            [10.0, 10.0],
            [1.0e-15, 1e-15],
            0.3,
            10000.0,
            1.5e-10,
            1.0e-3,
            True,
        ),
    ],
)
def test_cdm_parallel(
    ccd_5x5: CCD,
    input_array: np.ndarray,
    direction: Literal["parallel", "serial"],
    beta: float,
    trap_release_times: Sequence[float],
    trap_densities: Sequence[float],
    sigma: Sequence[float],
    full_well_capacity: float | None,
    max_electron_volume: float,
    transfer_period: float,
    charge_injection: bool,
):
    """Test function 'cdm' with valid inputs."""
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [999.99999999, 999.99999999, 999.99999999, 999.99999999, 999.99999999],
            [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        ]
    )
    detector = ccd_5x5
    detector.pixel.non_volatile.array = input_array

    cdm(
        detector=detector,
        direction=direction,
        trap_release_times=trap_release_times,
        trap_densities=trap_densities,
        sigma=sigma,
        beta=beta,
        full_well_capacity=full_well_capacity,
        max_electron_volume=max_electron_volume,
        transfer_period=transfer_period,
        charge_injection=charge_injection,
    )

    np.testing.assert_array_almost_equal(detector.pixel.array, expected)


def test_model_cdm_parallel(ccd_5x5: CCD):
    pixel_2d = np.array(
        [
            [47151.0, 50709.0, 43162.0, 52592.0, 49057.0],
            [46403.0, 50111.0, 49019.0, 57831.0, 45687.0],
            [44410.0, 51329.0, 51965.0, 45841.0, 46208.0],
            [49097.0, 51927.0, 49963.0, 57601.0, 52459.0],
            [49765.0, 48042.0, 52507.0, 44827.0, 38582.0],
        ],
        dtype=float,
    )

    detector = ccd_5x5
    detector.pixel.non_volatile.array = pixel_2d

    cdm(
        detector=detector,
        direction="parallel",
        trap_release_times=[3.0e-2, 0.5e-1],
        trap_densities=[20.0, 100.0],
        sigma=[1.0e-10, 1e-10],
        beta=0.3,
        max_electron_volume=1.62e-10,
        transfer_period=9.4722e-04,
        charge_injection=False,
    )

    expected = np.array(
        [
            [47151.0, 50709.0, 43162.0, 52592.0, 49057.0],
            [46402.9999999, 50110.9999999, 49018.9999999, 57830.9999999, 45686.9999999],
            [44409.9999999, 51328.9999999, 51964.9999999, 45840.9999999, 46207.9999999],
            [49096.9999999, 51926.9999999, 49962.9999999, 57600.9999999, 52458.9999999],
            [
                49764.9999999,
                48041.9999999,
                52506.9999999,
                44826.9999999,
                38581.99999998,
            ],
        ]
    )

    new_pixel = detector.pixel.array
    np.testing.assert_array_almost_equal(new_pixel, expected)


@pytest.mark.parametrize(
    "with_numba",
    [
        pytest.param(True, id="With Numba"),
        pytest.param(False, id="Without Numba"),
    ],
)
def test_run_cdm_parallel(with_numba: bool):
    pixel_2d = np.array(
        [
            [47151.0, 50709.0, 43162.0, 52592.0, 49057.0],
            [46403.0, 50111.0, 49019.0, 57831.0, 45687.0],
            [44410.0, 51329.0, 51965.0, 45841.0, 46208.0],
            [49097.0, 51927.0, 49963.0, 57601.0, 52459.0],
            [49765.0, 48042.0, 52507.0, 44827.0, 38582.0],
        ],
    )

    electron_effective_mass = 0.5
    full_well_capacity = 100_000  # unit: electron
    temperature = 273.15  # unit: K
    max_electron_volume = 1.62e-10  # unit: cm3
    transfer_period = 0.00094722  # unit: s
    beta = 0.3
    trap_release_times = np.array([0.03, 0.05])  # unit: s
    trap_densities = np.array([20.0, 100.0])  # unit: electron
    sigma = np.array([1e-10, 1e-10])  # unit: cm2 / electron

    e_effective_mass = electron_effective_mass * constants.m_e.value

    # Use factor 100 to convert to m/s to cm/s
    e_thermal_velocity = 100.0 * np.sqrt(
        3 * constants.k_B.value * temperature / e_effective_mass
    )

    if with_numba:
        # Test with 'numba'
        new_pixel_2d = run_cdm_parallel(
            array=pixel_2d,
            vg=max_electron_volume,
            t=transfer_period,
            fwc=full_well_capacity,
            vth=e_thermal_velocity,
            charge_injection=False,
            chg_inj_parallel_transfers=5,
            beta=beta,
            tr=trap_release_times,
            nt=trap_densities,
            sigma=sigma,
        )
    else:
        # Test without 'numba'
        new_pixel_2d = run_cdm_parallel.py_func(
            array=pixel_2d,
            vg=max_electron_volume,
            t=transfer_period,
            fwc=full_well_capacity,
            vth=e_thermal_velocity,
            charge_injection=False,
            chg_inj_parallel_transfers=5,
            beta=beta,
            tr=trap_release_times,
            nt=trap_densities,
            sigma=sigma,
        )

    expected = np.array(
        [
            [47151.0, 50709.0, 43162.0, 52592.0, 49057.0],
            [46402.9999999, 50110.9999999, 49018.9999999, 57830.9999999, 45686.9999999],
            [44409.9999999, 51328.9999999, 51964.9999999, 45840.9999999, 46207.9999999],
            [49096.9999999, 51926.9999999, 49962.9999999, 57600.9999999, 52458.9999999],
            [49764.9999999, 48041.9999999, 52506.9999999, 44826.9999999, 38581.9999999],
        ],
    )

    assert isinstance(new_pixel_2d, np.ndarray)
    np.testing.assert_array_almost_equal(np.asarray(new_pixel_2d), np.asarray(expected))


@pytest.mark.parametrize(
    """
    direction,
    trap_release_times,
    trap_densities,
    sigma,
    beta,
    full_well_capacity,
    max_electron_volume,
    transfer_period,
    charge_injection,
    """,
    [
        pytest.param(
            "serial",
            [5.0e-3, 5.0e-3],
            [10.0, 10.0],
            [1.0e-15, 1e-15],
            0.3,
            10000.0,
            1.5e-10,
            1.0e-3,
            None,
        ),
    ],
)
def test_cdm_serial(
    ccd_5x5: CCD,
    input_array: np.ndarray,
    direction: Literal["parallel", "serial"],
    beta: float,
    trap_release_times: Sequence[float],
    trap_densities: Sequence[float],
    sigma: Sequence[float],
    full_well_capacity: float | None,
    max_electron_volume: float,
    transfer_period: float,
    charge_injection: bool,
):
    """Test function 'cdm' with valid inputs."""
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        ]
    )
    detector = ccd_5x5
    detector.pixel.non_volatile.array = input_array

    cdm(
        detector=detector,
        direction=direction,
        trap_release_times=trap_release_times,
        trap_densities=trap_densities,
        sigma=sigma,
        beta=beta,
        full_well_capacity=full_well_capacity,
        max_electron_volume=max_electron_volume,
        transfer_period=transfer_period,
        charge_injection=charge_injection,
    )

    np.testing.assert_array_almost_equal(detector.pixel.array, expected)
