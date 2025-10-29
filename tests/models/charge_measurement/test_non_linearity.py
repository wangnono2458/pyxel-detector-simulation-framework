#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import pytest

from pyxel.detectors import (
    CCD,
    CMOS,
    CCDGeometry,
    Characteristics,
    CMOSGeometry,
    Environment,
    ReadoutProperties,
)
from pyxel.models.charge_measurement import (
    output_node_linearity_poly,
    physical_non_linearity,
    physical_non_linearity_with_saturation,
    simple_physical_non_linearity,
)
from pyxel.models.charge_measurement.non_linearity_calculation import ni_hansen


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
        environment=Environment(temperature=80),
        characteristics=Characteristics(),
    )
    detector._readout_properties = ReadoutProperties(times=[1.0])
    return detector


@pytest.fixture
def cmos_5x5() -> CMOS:
    """Create a valid CMOS detector."""
    detector = CMOS(
        geometry=CMOSGeometry(
            row=5,
            col=5,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(temperature=80),
        characteristics=Characteristics(),
    )
    detector._readout_properties = ReadoutProperties(times=[1.0, 2.0])
    return detector


@pytest.mark.parametrize(
    "x_cd, temperature, ctx_manager",
    [
        pytest.param(0.2, 4.0, nullcontext(), id="valid 1"),
        pytest.param(0.6, 4.0, nullcontext(), id="valid 2"),
        pytest.param(0.2, 300.0, nullcontext(), id="valid 3"),
        pytest.param(0.6, 300.0, nullcontext(), id="valid 4"),
        pytest.param(
            0.19,
            4.0,
            pytest.raises(ValueError, match="x_cd must be between 0.2 and 0.6"),
            id="'x_cd' too low",
        ),
        pytest.param(
            0.61,
            4.0,
            pytest.raises(ValueError, match="x_cd must be between 0.2 and 0.6"),
            id="'x_cd' too high",
        ),
    ],
)
def test_ni_hansen(x_cd, temperature, ctx_manager):
    """Test function 'ni_hansen'."""
    with ctx_manager:
        _ = ni_hansen(x_cd=x_cd, temperature=temperature)


@pytest.mark.parametrize(
    "coefficients",
    [[0, 1, 0.9], [5, 0.5, 0.9, 0.8], [3], [0, 3], [0, 1, -0.5]],
)
def test_non_linearity_valid(ccd_5x5: CCD, coefficients: Sequence):
    """Test model 'non_linearity' with valid inputs."""
    detector = ccd_5x5
    detector.signal.array = np.array(
        [
            [0.09, 0.05, 0.04, 0.06, 0.06],
            [0.05, 0.17, 0.06, 0.08, 0.05],
            [0.07, 0.09, 0.15, 0.07, 0.11],
            [2.04, 0.07, 0.06, 0.14, 0.1],
            [0.06, 0.05, 0.12, 0.04, 0.21],
        ]
    )
    output_node_linearity_poly(detector=detector, coefficients=coefficients)


@pytest.mark.parametrize(
    "coefficients, exp_exc, exp_error",
    [pytest.param([], ValueError, "Length of coefficient list should be more than 0.")],
)
def test_non_linearity_invalid(
    ccd_5x5: CCD, coefficients: Sequence, exp_exc, exp_error
):
    """Test model 'non_linearity' with valid inputs."""
    detector = ccd_5x5
    detector.signal.array = np.ones(detector.signal.shape)
    with pytest.raises(exp_exc, match=exp_error):
        output_node_linearity_poly(detector=detector, coefficients=coefficients)


def test_simple_physical_non_linearity_valid(cmos_5x5: CMOS):
    """Test model 'simple_physical_non_linearity' with valid inputs."""
    detector = cmos_5x5
    detector.signal.array = np.ones(detector.signal.shape)
    simple_physical_non_linearity(
        detector=detector,
        cutoff=2.0,
        n_acceptor=1.0e18,
        n_donor=1.0e15,
        diode_diameter=5.0,
        v_bias=0.1,
    )


@pytest.mark.parametrize(
    "temperature, ctx_manager",
    [
        pytest.param(
            3.9,
            pytest.raises(ValueError, match="temperature must be between 4K and 300K"),
            id="3.9 K",
        ),
        pytest.param(
            4.0,
            pytest.raises(
                ValueError, match="Intrinsic carrier concentration is equal to zero"
            ),
            id="4 K",
        ),
        pytest.param(
            5.0,
            pytest.raises(
                ValueError, match="Intrinsic carrier concentration is equal to zero"
            ),
            id="5 K",
        ),
        pytest.param(
            10.0,
            pytest.raises(
                ValueError, match="Intrinsic carrier concentration is equal to zero"
            ),
            id="10 K",
        ),
        pytest.param(100.0, nullcontext(), id="100 K"),
        pytest.param(300.0, nullcontext(), id="300 K"),
        pytest.param(
            300.1,
            pytest.raises(ValueError, match="temperature must be between 4K and 300K"),
            id="300.1 K",
        ),
    ],
)
def test_simple_physical_non_linearity_with_temperature(
    cmos_5x5: CCD, temperature: float, ctx_manager
):
    """Test model 'physical_non_linearity' with a 'CCD'."""
    assert isinstance(temperature, float)
    detector = cmos_5x5

    detector.pixel.non_volatile.array = np.ones(detector.signal.shape)
    detector.environment.temperature = temperature

    with ctx_manager:
        simple_physical_non_linearity(
            detector=detector,
            cutoff=2.0,
            n_acceptor=1.0e18,
            n_donor=1.0e15,
            diode_diameter=5.0,
            v_bias=0.1,
        )


def test_physical_non_linearity_valid(cmos_5x5: CMOS):
    """Test model 'physical_non_linearity' with valid inputs."""
    detector = cmos_5x5
    detector.pixel.non_volatile.array = np.ones(detector.signal.shape)
    physical_non_linearity(
        detector=detector,
        cutoff=2.0,
        n_acceptor=1.0e18,
        n_donor=1.0e15,
        diode_diameter=5.0,
        v_bias=0.1,
        fixed_capacitance=5.0e-15,
    )


def test_physical_non_linearity_with_saturation_valid(cmos_5x5: CMOS):
    """Test model 'physical_non_linearity_with_saturation' with valid inputs."""
    detector = cmos_5x5
    detector.signal.array = np.ones(detector.signal.shape)
    detector.photon.array = np.ones(detector.signal.shape)
    physical_non_linearity_with_saturation(
        detector=detector,
        cutoff=2.0,
        n_acceptor=1.0e18,
        n_donor=1.0e15,
        phi_implant=5.0,
        d_implant=2.0,
        saturation_current=0.001,
        ideality_factor=1.34,
        v_reset=0.0,
        d_sub=0.220,
        fixed_capacitance=5.0e-15,
        euler_points=100,
    )


@pytest.mark.parametrize(
    "temperature, ctx_manager",
    [
        pytest.param(
            3.9,
            pytest.raises(ValueError, match="temperature must be between 4K and 300K"),
            id="3.9 K",
        ),
        pytest.param(
            4.0,
            pytest.raises(
                ValueError, match="Intrinsic carrier concentration is equal to zero"
            ),
            marks=pytest.mark.xfail(reason="Bug ! Fix this"),
            id="4 K",
        ),
        pytest.param(
            5.0,
            pytest.raises(
                ValueError, match="Intrinsic carrier concentration is equal to zero"
            ),
            marks=pytest.mark.xfail(reason="Bug ! Fix this"),
            id="5 K",
        ),
        pytest.param(
            10.0,
            pytest.raises(
                ValueError, match="Intrinsic carrier concentration is equal to zero"
            ),
            marks=pytest.mark.xfail(reason="Bug ! Fix this"),
            id="10 K",
        ),
        pytest.param(100.0, nullcontext(), id="100 K"),
        pytest.param(300.0, nullcontext(), id="300 K"),
        pytest.param(
            300.1,
            pytest.raises(ValueError, match="temperature must be between 4K and 300K"),
            id="300.1 K",
        ),
    ],
)
def test_physical_non_linearity_with_saturation_with_temperature(
    cmos_5x5: CCD, temperature: float, ctx_manager
):
    detector = cmos_5x5
    detector.environment.temperature = temperature

    detector.signal.array = np.ones(detector.signal.shape)
    detector.photon.array = np.ones(detector.signal.shape)

    with ctx_manager:
        physical_non_linearity_with_saturation(
            detector=detector,
            cutoff=2.0,
            n_acceptor=1.0e18,
            n_donor=1.0e15,
            phi_implant=5.0,
            d_implant=2.0,
            saturation_current=0.001,
            ideality_factor=1.34,
            v_reset=0.0,
            d_sub=0.220,
            fixed_capacitance=5.0e-15,
            euler_points=100,
        )


def test_simple_physical_non_linearity_with_ccd(ccd_5x5: CCD):
    """Test model 'simple_physical_non_linearity' with a 'CCD'."""
    detector = ccd_5x5

    with pytest.raises(TypeError, match="Expecting a 'CMOS' detector object."):
        simple_physical_non_linearity(
            detector=detector,
            cutoff=2.0,
            n_acceptor=1.0e18,
            n_donor=1.0e15,
            diode_diameter=5.0,
            v_bias=0.1,
        )


def test_physical_non_linearity_with_saturation_multiple_readout(cmos_5x5: CCD):
    """Test model 'physical_non_linearity_with_saturation' with multiple readouts."""
    detector = cmos_5x5
    detector.signal.array = np.ones(detector.signal.shape)
    detector.photon.array = np.ones(detector.signal.shape)

    # First exposure
    detector.readout_properties.pipeline_count = 0
    physical_non_linearity_with_saturation(
        detector=detector,
        cutoff=2.0,
        n_acceptor=1.0e18,
        n_donor=1.0e15,
        phi_implant=5.0,
        d_implant=2.0,
        saturation_current=0.001,
        ideality_factor=1.34,
        v_reset=0.0,
        d_sub=0.220,
        fixed_capacitance=5.0e-15,
        euler_points=100,
    )

    # Second exposure
    detector.readout_properties.pipeline_count = 1
    physical_non_linearity_with_saturation(
        detector=detector,
        cutoff=2.0,
        n_acceptor=1.0e18,
        n_donor=1.0e15,
        phi_implant=5.0,
        d_implant=2.0,
        saturation_current=0.001,
        ideality_factor=1.34,
        v_reset=0.0,
        d_sub=0.220,
        fixed_capacitance=5.0e-15,
        euler_points=100,
    )


def test_physical_non_linearity_with_ccd(ccd_5x5: CCD):
    """Test model 'physical_non_linearity' with a 'CCD'."""
    detector = ccd_5x5

    with pytest.raises(TypeError, match="Expecting a 'CMOS' detector object."):
        physical_non_linearity(
            detector=detector,
            cutoff=2.0,
            n_acceptor=1.0e18,
            n_donor=1.0e15,
            diode_diameter=5.0,
            v_bias=0.1,
            fixed_capacitance=5.0e-15,
        )


@pytest.mark.parametrize(
    "temperature, ctx_manager",
    [
        pytest.param(
            3.9,
            pytest.raises(ValueError, match="temperature must be between 4K and 300K"),
            id="3.9 K",
        ),
        pytest.param(
            4.0,
            pytest.raises(
                ValueError, match="Intrinsic carrier concentration is equal to zero"
            ),
            id="4 K",
        ),
        pytest.param(
            5.0,
            pytest.raises(
                ValueError, match="Intrinsic carrier concentration is equal to zero"
            ),
            id="5 K",
        ),
        pytest.param(
            10.0,
            pytest.raises(
                ValueError, match="Intrinsic carrier concentration is equal to zero"
            ),
            id="10 K",
        ),
        pytest.param(100.0, nullcontext(), id="100 K"),
        pytest.param(300.0, nullcontext(), id="300 K"),
        pytest.param(
            300.1,
            pytest.raises(ValueError, match="temperature must be between 4K and 300K"),
            id="300.1 K",
        ),
    ],
)
def test_physical_non_linearity_with_temperature(
    cmos_5x5: CCD, temperature: float, ctx_manager
):
    """Test model 'physical_non_linearity' with a 'CCD'."""
    assert isinstance(temperature, float)
    detector = cmos_5x5

    detector.pixel.non_volatile.array = np.ones(detector.signal.shape)
    detector.environment.temperature = temperature

    with ctx_manager:
        physical_non_linearity(
            detector=detector,
            cutoff=2.0,
            n_acceptor=1.0e18,
            n_donor=1.0e15,
            diode_diameter=5.0,
            v_bias=0.1,
            fixed_capacitance=5.0e-15,
        )


def test_physical_non_linearity_with_saturation_with_ccd(ccd_5x5: CCD):
    """Test model 'physical_non_linearity_with_saturation' with a 'CCD'."""
    detector = ccd_5x5

    with pytest.raises(TypeError, match="Expecting a 'CMOS' detector object."):
        physical_non_linearity_with_saturation(
            detector=detector,
            cutoff=2.0,
            n_acceptor=1.0e18,
            n_donor=1.0e15,
            phi_implant=5.0,
            d_implant=2.0,
            saturation_current=0.001,
            ideality_factor=1.34,
            v_reset=0.0,
            d_sub=0.220,
            fixed_capacitance=5.0e-15,
            euler_points=100,
        )
