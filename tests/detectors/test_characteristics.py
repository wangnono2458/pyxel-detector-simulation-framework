#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import numpy as np
import pytest

from pyxel.detectors import CCD, CCDGeometry, Channels, Characteristics, Environment
from pyxel.detectors.channels import Matrix, ReadoutPosition


@pytest.mark.parametrize(
    "quantum_efficiency, charge_to_volt_conversion, pre_amplification, "
    "full_well_capacity, adc_voltage_range, adc_bit_resolution",
    [
        (None, None, None, None, None, None),
        (0.0, 0, 0, 0, None, None),
        (1.0, 1.0, 100.0, 10_000_000, None, None),
        (0.5, 1.0, 100.0, 10_000_000, [0.0, 3.0], 16),
    ],
)
def test_characteristics(
    quantum_efficiency,
    charge_to_volt_conversion,
    pre_amplification,
    full_well_capacity,
    adc_voltage_range,
    adc_bit_resolution,
):
    """Test 'Characteristics.__init__'."""
    obj = Characteristics(
        quantum_efficiency=quantum_efficiency,
        charge_to_volt_conversion=charge_to_volt_conversion,
        pre_amplification=pre_amplification,
        full_well_capacity=full_well_capacity,
        adc_voltage_range=adc_voltage_range,
        adc_bit_resolution=adc_bit_resolution,
    )

    # Test property 'numbytes'
    _ = obj.numbytes

    # Test getter 'Characteristics.quantum_efficiency'
    if quantum_efficiency is None:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'quantum_efficiency' in 'characteristics'",
        ):
            _ = obj.quantum_efficiency
    else:
        assert obj.quantum_efficiency == quantum_efficiency

    # Test getter 'Characteristics.charge_to_volt_conversion'
    if charge_to_volt_conversion is None:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'charge_to_volt_conversion' in 'characteristics'",
        ):
            _ = obj.charge_to_volt_conversion
    else:
        assert obj.charge_to_volt_conversion == charge_to_volt_conversion

    # Test getter 'Characteristics.pre_amplification'
    if pre_amplification is None:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'pre_amplification' in 'characteristics'",
        ):
            _ = obj.pre_amplification
    else:
        assert obj.pre_amplification == pre_amplification

    # Test getter 'Characteristics.adc_bit_resolution'
    if adc_bit_resolution is None:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'adc_bit_resolution' in 'characteristics'",
        ):
            _ = obj.adc_bit_resolution
    else:
        assert obj.adc_bit_resolution == adc_bit_resolution

    # Test getter 'Characteristics.adc_voltage_range'
    if adc_voltage_range is None:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'adc_voltage_range' in 'characteristics'",
        ):
            _ = obj.adc_voltage_range
    else:
        assert obj.adc_voltage_range == tuple(adc_voltage_range)

    # Test getter 'Characteristics.full_well_capacity'
    if full_well_capacity is None:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'full_well_capacity' in 'characteristics'",
        ):
            _ = obj.full_well_capacity
    else:
        assert obj.full_well_capacity == full_well_capacity


@pytest.mark.parametrize(
    "quantum_efficiency, charge_to_volt_conversion, pre_amplification, "
    "full_well_capacity, adc_voltage_range, adc_bit_resolution, "
    "exp_exc, exp_msg",
    [
        # (0., 0, 0, 0, None, None, ValueError,'foo'),
        (
            -0.1,
            0,
            0,
            0,
            None,
            None,
            ValueError,
            r"'quantum_efficiency' must be between 0.0 and 1.0.",
        ),
        (
            -1.1,
            0,
            0,
            0,
            None,
            None,
            ValueError,
            r"'quantum_efficiency' must be between 0.0 and 1.0.",
        ),
        (
            0.0,
            -0.1,
            0,
            0,
            None,
            None,
            ValueError,
            r"'charge_to_volt_conversion' must be between 0.0 and 100.0.",
        ),
        (
            0.0,
            100.1,
            0,
            0,
            None,
            None,
            ValueError,
            r"'charge_to_volt_conversion' must be between 0.0 and 100.0.",
        ),
        (
            0.0,
            0,
            -0.1,
            0,
            None,
            None,
            ValueError,
            r"'pre_amplification' must be between 0.0 and 10000.0.",
        ),
        (
            0.0,
            0,
            10_000.1,
            0,
            None,
            None,
            ValueError,
            r"'pre_amplification' must be between 0.0 and 10000.0.",
        ),
        (
            0.0,
            0,
            0,
            -0.1,
            None,
            None,
            ValueError,
            "'full_well_capacity' must be between 0 and 1e7.",
        ),
        (
            0.0,
            0,
            0,
            10_000_000.1,
            None,
            None,
            ValueError,
            "'full_well_capacity' must be between 0 and 1e7.",
        ),
        (0.0, 0, 0, 0, 1.0, None, TypeError, r"Voltage range must have length of 2."),
        (
            0.0,
            0,
            0,
            0,
            (1.0, 2.0, 3.0),
            None,
            ValueError,
            r"Voltage range must have length of 2.",
        ),
        (
            0.0,
            0,
            0,
            0,
            None,
            3,
            ValueError,
            r"'adc_bit_resolution' must be between 4 and 64.",
        ),
        (
            0.0,
            0,
            0,
            0,
            None,
            65,
            ValueError,
            r"'adc_bit_resolution' must be between 4 and 64.",
        ),
    ],
)
def test_characteristics_invalid(
    quantum_efficiency,
    charge_to_volt_conversion,
    pre_amplification,
    full_well_capacity,
    adc_voltage_range,
    adc_bit_resolution,
    exp_exc: type[Exception],
    exp_msg: str,
):
    """Test 'Characteristics.__init__'."""
    with pytest.raises(exp_exc, match=exp_msg):
        _ = Characteristics(
            quantum_efficiency=quantum_efficiency,
            charge_to_volt_conversion=charge_to_volt_conversion,
            pre_amplification=pre_amplification,
            full_well_capacity=full_well_capacity,
            adc_voltage_range=adc_voltage_range,
            adc_bit_resolution=adc_bit_resolution,
        )


@pytest.mark.parametrize("quantum_efficiency", [0.0, 1.0])
def test_quantum_efficiency_setter(quantum_efficiency):
    """Test setter 'Characteristics.quantum_efficiency'."""
    obj = Characteristics()

    obj.quantum_efficiency = quantum_efficiency
    assert obj.quantum_efficiency == quantum_efficiency


@pytest.mark.parametrize(
    "quantum_efficiency, exp_exc, exp_msg",
    [
        (-0.1, ValueError, r"'quantum_efficiency' values must be between 0.0 and 1.0."),
        (1.1, ValueError, r"'quantum_efficiency' values must be between 0.0 and 1.0."),
    ],
)
def test_quantum_efficiency_setter_wrong_inputs(
    quantum_efficiency, exp_exc: type[Exception], exp_msg: str
):
    """Test setter 'Characteristics.quantum_efficiency'."""
    obj = Characteristics()

    with pytest.raises(exp_exc, match=exp_msg):
        obj.quantum_efficiency = quantum_efficiency


@pytest.mark.parametrize("charge_to_volt_conversion", [0.0, 100.0])
def test_charge_to_volt_conversion_setter(charge_to_volt_conversion):
    """Test setter 'Characteristics.charge_to_volt_conversion'."""
    obj = Characteristics()

    obj.charge_to_volt_conversion = charge_to_volt_conversion
    assert obj.charge_to_volt_conversion == charge_to_volt_conversion


def test_pre_amplification_with_channels():
    # Setup the detector with geometry and characteristics
    detector = CCD(
        geometry=CCDGeometry(
            row=4,
            col=4,
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
        characteristics=Characteristics(),
    )

    # Initialize detector with geometry to build channel gains
    detector.characteristics.initialize(detector.geometry)

    # Set charge_to_volt_conversion with dictionary values and validate
    detector.characteristics.pre_amplification = {
        "OP9": 1,
        "OP13": 2,
        "OP1": 3,
        "OP5": 4,
    }
    assert detector.characteristics.pre_amplification == {
        "OP9": 1,
        "OP13": 2,
        "OP1": 3,
        "OP5": 4,
    }

    # Assert that the channel gains have been set correctly
    expected_gain_matrix = np.array(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        detector.characteristics.channels_pre_amplification, expected_gain_matrix
    )

    # Change the values and validate again
    detector.characteristics.pre_amplification = {
        "OP9": 1.1,
        "OP13": 2.2,
        "OP1": 3.3,
        "OP5": 4.4,
    }
    assert detector.characteristics.pre_amplification == {
        "OP9": 1.1,
        "OP13": 2.2,
        "OP1": 3.3,
        "OP5": 4.4,
    }
    expected_gain_matrix = np.array(
        [
            [1.1, 1.1, 2.2, 2.2],
            [1.1, 1.1, 2.2, 2.2],
            [3.3, 3.3, 4.4, 4.4],
            [3.3, 3.3, 4.4, 4.4],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        detector.characteristics.channels_pre_amplification, expected_gain_matrix
    )


def test_pre_amplification_invalid_values():
    detector = CCD(
        geometry=CCDGeometry(
            row=4,
            col=4,
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
        characteristics=Characteristics(),
    )

    # Initialize detector with geometry to build channel gains
    detector.characteristics.initialize(detector.geometry)

    # Attempt to set charge_to_volt_conversion with invalid dictionary values
    with pytest.raises(ValueError, match=r"\'pre_amplification\' must be between"):
        detector.characteristics.pre_amplification = {
            "OP9": 1,  # Valid
            "OP13": 200,  # Invalid: Outside the range 0 to 100
            "OP1": 3,  # Valid
            "OP5": -10,  # Invalid: Negative value
        }


def test_channel_gain_mismatch():
    # Setup the detector with a discrepancy in the matrix channel count
    detector = CCD(
        geometry=CCDGeometry(
            row=4,
            col=4,
            channels=Channels(
                matrix=Matrix(
                    [["OP9", "OP13"], ["OP1", "OP5"]]
                ),  # Only two channels are defined here
                readout_position=ReadoutPosition(
                    {
                        "OP9": "top-left",
                        "OP13": "top-left",
                        "OP1": "top-right",  # Valid
                        "OP5": "top-right",  # Invalid: Negative value
                    }
                ),
            ),
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )

    detector.characteristics.initialize(detector.geometry)

    # Intentionally provide mismatched number of channel gains
    with pytest.raises(
        ValueError, match=r"Mismatch between the defined channels"
    ) as exc_info:
        detector.characteristics.pre_amplification = {"OP9": 1, "OP13": 2}

    # Check if the correct error message about mismatched channel counts is raised
    assert (
        "Mismatch between the defined channels in geometry and provided channel gains"
        in str(exc_info.value)
    )


@pytest.mark.parametrize(
    "charge_to_volt_conversion, exp_exc, exp_msg",
    [
        (
            -0.2,
            ValueError,
            r"'charge_to_volt_conversion' must be between 0.0 and 100.0.",
        ),
        (
            100.2,
            ValueError,
            r"'charge_to_volt_conversion' must be between 0.0 and 100.0.",
        ),
    ],
)
def test_charge_to_volt_conversion_setter_wrong_inputs(
    charge_to_volt_conversion, exp_exc: type[Exception], exp_msg: str
):
    """Test setter 'Characteristics.charge_to_volt_conversion'."""
    obj = Characteristics()

    with pytest.raises(exp_exc, match=exp_msg):
        obj.charge_to_volt_conversion = charge_to_volt_conversion


@pytest.mark.parametrize("pre_amplification", [0.0, 10_000.0])
def test_pre_amplification_setter(pre_amplification):
    """Test setter 'Characteristics.pre_amplification'."""
    obj = Characteristics()

    obj.pre_amplification = pre_amplification
    assert obj.pre_amplification == pre_amplification


@pytest.mark.parametrize(
    "pre_amplification, exp_exc, exp_msg",
    [
        (-0.1, ValueError, r"'pre_amplification' must be between 0.0 and 10000.0."),
        (10_000.1, ValueError, r"'pre_amplification' must be between 0.0 and 10000.0."),
    ],
)
def test_pre_amplification_setter_wrong_inputs(
    pre_amplification, exp_exc: type[Exception], exp_msg: str
):
    """Test setter 'Characteristics.pre_amplification'."""
    obj = Characteristics()

    with pytest.raises(exp_exc, match=exp_msg):
        obj.pre_amplification = pre_amplification


@pytest.mark.parametrize("full_well_capacity", [0.0, 10_000.0])
def test_full_well_capacity(full_well_capacity):
    """Test setter 'Characteristics.full_well_capacity'."""
    obj = Characteristics()

    obj.full_well_capacity = full_well_capacity
    assert obj.full_well_capacity == full_well_capacity


@pytest.mark.parametrize(
    "full_well_capacity, exp_exc, exp_msg",
    [
        (-0.1, ValueError, r"'full_well_capacity' must be between"),
        (10_000_000.1, ValueError, r"'full_well_capacity' must be between"),
    ],
)
def test_full_well_capacity_setter_wrong_inputs(
    full_well_capacity, exp_exc: type[Exception], exp_msg: str
):
    """Test setter 'Characteristics.full_well_capacity'."""
    obj = Characteristics()

    with pytest.raises(exp_exc, match=exp_msg):
        obj.full_well_capacity = full_well_capacity


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(Characteristics(), False, id="Empty 'Characteristics'"),
        pytest.param(
            Characteristics(quantum_efficiency=0.1), False, id="Only one parameter"
        ),
        pytest.param(
            Characteristics(
                quantum_efficiency=0.1,
                charge_to_volt_conversion=0.2,
                pre_amplification=4.4,
            ),
            False,
            id="Wrong type",
        ),
        pytest.param(
            Characteristics(
                quantum_efficiency=0.1,
                charge_to_volt_conversion=0.2,
                pre_amplification=4.4,
                full_well_capacity=10,
            ),
            False,
            id="Missing some parameters",
        ),
        pytest.param(
            Characteristics(
                quantum_efficiency=0.1,
                charge_to_volt_conversion=0.2,
                pre_amplification=4.4,
                full_well_capacity=10,
                adc_voltage_range=(0.0, 10.0),
                adc_bit_resolution=16,
            ),
            True,
            id="Same parameters, same class",
        ),
    ],
)
def test_is_equal(other_obj, is_equal):
    """Test equality statement for `Characteristics`."""
    obj = Characteristics(
        quantum_efficiency=0.1,
        charge_to_volt_conversion=0.2,
        pre_amplification=4.4,
        full_well_capacity=10,
        adc_voltage_range=(0.0, 10.0),
        adc_bit_resolution=16,
    )

    if is_equal:
        assert obj == other_obj
    else:
        assert obj != other_obj


@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (
            Characteristics(
                quantum_efficiency=0.1,
                charge_to_volt_conversion=0.2,
                pre_amplification=4.4,
                full_well_capacity=10,
            ),
            {
                "quantum_efficiency": 0.1,
                "charge_to_volt_conversion": 0.2,
                "pre_amplification": 4.4,
                "full_well_capacity": 10,
                "adc_bit_resolution": None,
                "adc_voltage_range": None,
            },
        ),
        (
            Characteristics(
                quantum_efficiency=0.1,
                charge_to_volt_conversion=0.2,
                pre_amplification=4.4,
                full_well_capacity=10,
                adc_voltage_range=(0.0, 10.0),
                adc_bit_resolution=16,
            ),
            {
                "quantum_efficiency": 0.1,
                "charge_to_volt_conversion": 0.2,
                "pre_amplification": 4.4,
                "full_well_capacity": 10,
                "adc_voltage_range": (0.0, 10.0),
                "adc_bit_resolution": 16,
            },
        ),
    ],
)
def test_to_and_from_dict(obj, exp_dict):
    """Test methods 'to_dict', 'from_dict'."""
    assert type(obj) == Characteristics

    # Convert from `Characteristics` to a `dict`
    dct = obj.to_dict()
    assert dct == exp_dict

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    assert copied_dct == exp_dict

    # Convert from `dict` to `Characteristics`
    other_obj = Characteristics.from_dict(copied_dct)
    assert type(other_obj) == Characteristics
    assert obj == other_obj
    assert obj is not other_obj
    assert copied_dct == exp_dict
