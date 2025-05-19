#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import pytest

from pyxel.detectors import Environment, WavelengthHandling


@pytest.mark.parametrize("temperature", [None, 0.1, 1000.0])
@pytest.mark.parametrize(
    "wavelength",
    [None, 1.0, 1000.0, WavelengthHandling(cut_on=1.0, cut_off=5.0, resolution=2)],
)
def test_environment(temperature, wavelength):
    """Test when creating a valid `Environment` object."""
    obj = Environment(temperature=temperature, wavelength=wavelength)

    # Test __repr__
    assert repr(obj).startswith("Environment(")

    # Test property 'numbytes'
    _ = obj.numbytes

    # Test getter 'Environment.temperature'
    if temperature is None:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'temperature' in 'environment'",
        ):
            _ = obj.temperature
    else:
        assert obj.temperature == temperature

    # Test getter 'Environment.wavelength'
    if wavelength is None:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'wavelength' in 'environment'",
        ):
            _ = obj.wavelength
    else:
        assert obj.wavelength == wavelength


@pytest.mark.parametrize(
    "temperature, wavelength, exp_exc, exp_error",
    [
        pytest.param(-0.1, None, ValueError, r"\'temperature\' must be between"),
        pytest.param(1001, None, ValueError, r"\'temperature\' must be between"),
        pytest.param(None, 0, ValueError, r"'wavelength' must be strictly positive"),
        pytest.param(None, -1.0, ValueError, r"'wavelength' must be strictly positive"),
    ],
)
def test_invalid_environment(
    temperature: float | None,
    wavelength: float | None,
    exp_exc: type[Exception],
    exp_error: str,
):
    """Test when creating an invalid `Environment` object."""
    with pytest.raises(exp_exc, match=exp_error):
        _ = Environment(temperature=temperature, wavelength=wavelength)


@pytest.mark.parametrize("temperature", [0.1, 1000.0])
def test_temperature_setter(temperature):
    """Test setter 'Environment.temperature'."""
    obj = Environment()

    obj.temperature = temperature
    assert obj.temperature == temperature


@pytest.mark.parametrize(
    "temperature, exp_exc, exp_msg",
    [
        (-1, ValueError, r"'temperature' must be between 0.0 and 1000.0"),
        (0.0, ValueError, r"'temperature' must be between 0.0 and 1000.0"),
        (1001.0, ValueError, r"'temperature' must be between 0.0 and 1000.0"),
    ],
)
def test_temperature_setter_wrong_inputs(
    temperature, exp_exc: type[Exception], exp_msg: str
):
    """Test setter 'Environment.temperature'."""
    obj = Environment()

    with pytest.raises(exp_exc, match=exp_msg):
        obj.temperature = temperature


@pytest.mark.parametrize(
    "wavelength",
    [0.1, 1000.0, WavelengthHandling(cut_on=1.0, cut_off=5.0, resolution=2)],
)
def test_wavelength_setter(wavelength):
    """Test setter 'Environment.wavelength'."""
    obj = Environment()

    obj.wavelength = wavelength
    assert obj.wavelength == wavelength


@pytest.mark.parametrize(
    "wavelength, exp_exc, exp_msg",
    [
        (-1, ValueError, r"'wavelength' must be strictly positive"),
        (0.0, ValueError, r"'wavelength' must be strictly positive"),
        (None, TypeError, r"A WavelengthHandling object or a float must be provided"),
    ],
)
def test_wavelength_setter_wrong_inputs(
    wavelength, exp_exc: type[Exception], exp_msg: str
):
    """Test setter 'Environment.wavelength'."""
    obj = Environment()

    with pytest.raises(exp_exc, match=exp_msg):
        obj.wavelength = wavelength


@pytest.mark.parametrize(
    "other_obj, is_equal",
    [
        pytest.param(None, False, id="None"),
        pytest.param(Environment(), False, id="Empty 'Environment'"),
        pytest.param(Environment(temperature=100.1), True, id="Valid"),
    ],
)
def test_is_equal(other_obj, is_equal):
    """Test equality statement for Environment."""
    obj = Environment(temperature=100.1)

    if is_equal:
        assert obj == other_obj
    else:
        assert obj != other_obj


@pytest.mark.parametrize(
    "obj, exp_dict",
    [
        (Environment(), {"temperature": None}),
        (Environment(temperature=100.1), {"temperature": 100.1}),
        (Environment(wavelength=200.0), {"temperature": None, "wavelength": 200.0}),
        (
            Environment(temperature=100.1, wavelength=200.0),
            {"temperature": 100.1, "wavelength": 200.0},
        ),
        (
            Environment(
                temperature=100.1,
                wavelength=WavelengthHandling(cut_on=1.0, cut_off=5.0, resolution=2),
            ),
            {
                "temperature": 100.1,
                "wavelength": {"cut_on": 1.0, "cut_off": 5.0, "resolution": 2},
            },
        ),
    ],
)
def test_to_and_from_dict(obj, exp_dict):
    """Test methods 'to_dict' and 'from_dict'."""

    assert type(obj) == Environment

    # Convert from `Environment` to a `dict`
    dct = obj.to_dict()
    assert dct == exp_dict

    # Copy 'exp_dict'
    copied_dct = deepcopy(exp_dict)  # create a new dict
    assert copied_dct is not exp_dict
    assert copied_dct == exp_dict

    # Convert from `dict` to `Environment`
    other_obj = Environment.from_dict(copied_dct)
    assert type(other_obj) == Environment
    assert obj == other_obj
    assert obj is not other_obj
    assert copied_dct == exp_dict
