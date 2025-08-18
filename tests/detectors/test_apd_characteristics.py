#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass

import astropy.constants as const
import pytest

from pyxel.detectors import APDCharacteristics
from pyxel.detectors.apd import AvalancheSettings, ConverterFunction, ConverterValues


@dataclass
class InputParams:
    avalanche_gain: float | None = None
    pixel_reset_voltage: float | None = None
    common_voltage: float | None = None


@dataclass
class OutputParams:
    avalanche_gain: float
    pixel_reset_voltage: float
    common_voltage: float


@dataclass
class Params:
    input: InputParams
    output: OutputParams


@pytest.fixture(
    params=(
        "avalanche_gain_and_pixel_reset_voltage",
        "avalanche_gain_and_common_voltage",
        "common_voltage_and_pixel_reset_voltage",
    )
)
def valid_characteristics(request):
    if request.param == "avalanche_gain_and_pixel_reset_voltage":
        return Params(
            input=InputParams(avalanche_gain=1.0, pixel_reset_voltage=2.0),
            output=OutputParams(
                avalanche_gain=1.0, pixel_reset_voltage=2.0, common_voltage=-0.65
            ),
        )
    elif request.param == "avalanche_gain_and_common_voltage":
        return Params(
            input=InputParams(avalanche_gain=1.0, common_voltage=2.0),
            output=OutputParams(
                avalanche_gain=1.0, pixel_reset_voltage=4.65, common_voltage=2.0
            ),
        )
    elif request.param == "common_voltage_and_pixel_reset_voltage":
        return Params(
            input=InputParams(common_voltage=2.0, pixel_reset_voltage=3.0),
            output=OutputParams(
                avalanche_gain=1.0, pixel_reset_voltage=3.0, common_voltage=2.0
            ),
        )
    else:
        raise NotImplementedError


@pytest.fixture(
    params=(
        "avalanche_gain_and_pixel_reset_voltage",
        "avalanche_gain_and_common_voltage",
    )
)
def valid_characteristics_only_avalanche(request):
    if request.param == "avalanche_gain_and_pixel_reset_voltage":
        return Params(
            input=InputParams(avalanche_gain=1.0, pixel_reset_voltage=2.0),
            output=OutputParams(
                avalanche_gain=1.0, pixel_reset_voltage=2.0, common_voltage=-0.65
            ),
        )
    elif request.param == "avalanche_gain_and_common_voltage":
        return Params(
            input=InputParams(avalanche_gain=1.0, common_voltage=2.0),
            output=OutputParams(
                avalanche_gain=1.0, pixel_reset_voltage=4.65, common_voltage=2.0
            ),
        )
    else:
        raise NotImplementedError


@pytest.mark.parametrize(
    "quantum_efficiency",
    [
        pytest.param(None, id="no QE"),
        pytest.param(0.0, id="QE == 0"),
        pytest.param(1.0, id="QE == 1"),
    ],
)
@pytest.mark.parametrize(
    "adc_bit_resolution",
    [
        pytest.param(None, id="No ADC bits"),
        pytest.param(4, id="ADC 4 bits"),
        pytest.param(64, id="ADC 64 bits"),
    ],
)
@pytest.mark.parametrize(
    "adc_voltage_range",
    [
        pytest.param(None, id="No ADC voltage range"),
        pytest.param((1.0, 2.0), id="With ADC voltage range"),
        pytest.param((2.0, 1.0), id="With ADC voltage range2"),
        pytest.param((1.0, 1.0), id="With ADC voltage range3"),
    ],
)
@pytest.mark.parametrize(
    "full_well_capacity",
    [
        pytest.param(None, id="No full_well_capacity"),
        pytest.param(0, id="full_well_capacity == 0"),
        pytest.param(10_000_000.0, id="full_well_capacity == 10_000_000"),
    ],
)
@pytest.mark.parametrize("constructor", ["init", "build"])
def test_constructors(
    valid_characteristics: Params,
    quantum_efficiency,
    adc_bit_resolution,
    adc_voltage_range,
    full_well_capacity,
    constructor: str,
):
    """Test methods'APDCharacteristics.__init__' and '.build'."""

    if constructor == "init":
        obj = APDCharacteristics(
            roic_gain=0.5,
            bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
            avalanche_settings=AvalancheSettings(
                avalanche_gain=valid_characteristics.input.avalanche_gain,
                pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
                common_voltage=valid_characteristics.input.common_voltage,
                gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
            ),
            quantum_efficiency=quantum_efficiency,
            adc_bit_resolution=adc_bit_resolution,
            adc_voltage_range=adc_voltage_range,
            full_well_capacity=full_well_capacity,
        )

    elif constructor == "build":
        dct = {
            "roic_gain": 0.5,
            "bias_to_node": {
                "values": [  # <-- Dummy node capacitance input
                    (2.65, 73.7),
                    (4.0, 60.0),
                ]
            },
            "avalanche_settings": {
                "avalanche_gain": valid_characteristics.input.avalanche_gain,
                "pixel_reset_voltage": valid_characteristics.input.pixel_reset_voltage,
                "common_voltage": valid_characteristics.input.common_voltage,
                "gain_to_bias": {"function": "lambda gain: 0.15 * gain + 2.5"},
                "bias_to_gain": {"values": [(2.65, 1.0), (4.0, 10.0)]},
            },
            "quantum_efficiency": quantum_efficiency,
            "adc_bit_resolution": adc_bit_resolution,
            "adc_voltage_range": adc_voltage_range,
            "full_well_capacity": full_well_capacity,
        }

        obj = APDCharacteristics.build(dct)
    else:
        raise NotImplementedError

    # Optional: add assertions
    assert isinstance(obj, APDCharacteristics)

    assert obj.avalanche_settings.avalanche_gain == pytest.approx(
        valid_characteristics.output.avalanche_gain
    )
    assert obj.avalanche_settings.pixel_reset_voltage == pytest.approx(
        valid_characteristics.output.pixel_reset_voltage
    )
    assert obj.avalanche_settings.common_voltage == pytest.approx(
        valid_characteristics.output.common_voltage
    )

    # Check 'quantum_efficiency'
    if quantum_efficiency is not None:
        assert obj.quantum_efficiency == quantum_efficiency
    else:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'quantum_efficiency' in 'characteristics'",
        ):
            _ = obj.quantum_efficiency

    # Check 'adc_bit_resolution'
    if adc_bit_resolution is not None:
        assert obj.adc_bit_resolution == adc_bit_resolution
    else:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'adc_bit_resolution' in 'characteristics'",
        ):
            _ = obj.adc_bit_resolution

    # Check 'adc_voltage_range'
    if adc_voltage_range is not None:
        assert obj.adc_voltage_range == adc_voltage_range
    else:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'adc_voltage_range' in 'characteristics'",
        ):
            _ = obj.adc_voltage_range

    # Check 'full_well_capacity'
    if full_well_capacity is not None:
        assert obj.full_well_capacity == full_well_capacity
    else:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter 'full_well_capacity' in 'characteristics'",
        ):
            _ = obj.full_well_capacity


@pytest.mark.parametrize(
    "quantum_efficiency",
    [
        pytest.param(-0.1, id="QE too low"),
        pytest.param(1.1, id="QE too high"),
    ],
)
def test_invalid_initialization_qe(valid_characteristics: Params, quantum_efficiency):
    with pytest.raises(
        ValueError, match="'quantum_efficiency' must be between 0\\.0 and 1\\.0"
    ):
        _ = APDCharacteristics(
            roic_gain=0.5,
            bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
            avalanche_settings=AvalancheSettings(
                avalanche_gain=valid_characteristics.input.avalanche_gain,
                pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
                common_voltage=valid_characteristics.input.common_voltage,
                gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
            ),
            quantum_efficiency=quantum_efficiency,
        )


@pytest.mark.parametrize(
    "avalanche_gain",
    [
        pytest.param(-0.1, id="'avalanche_gain' < 0"),
        pytest.param(0.9, id="'avalanche_gain' < 1"),
        pytest.param(1000.1, id="'avalanche_gain' > 1000"),
    ],
)
def test_invalid_initialization_avalanche_gain(
    valid_characteristics_only_avalanche: Params, avalanche_gain
):
    with pytest.raises(ValueError, match="Invalid 'avalanche_gain"):
        _ = APDCharacteristics(
            roic_gain=0.5,
            avalanche_settings=AvalancheSettings(
                avalanche_gain=avalanche_gain,
                pixel_reset_voltage=valid_characteristics_only_avalanche.input.pixel_reset_voltage,
                common_voltage=valid_characteristics_only_avalanche.input.common_voltage,
                gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
            ),
        )


@pytest.mark.parametrize(
    "adc_bit_resolution",
    [
        pytest.param(3, id="adc_bit_resolution too low"),
        pytest.param(65, id="adc_bit_resolution too high"),
    ],
)
def test_invalid_initialization_adc_bit_resolution(
    valid_characteristics: Params, adc_bit_resolution
):
    with pytest.raises(
        ValueError, match="'adc_bit_resolution' must be between 4 and 64"
    ):
        _ = APDCharacteristics(
            roic_gain=0.5,
            bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
            avalanche_settings=AvalancheSettings(
                avalanche_gain=valid_characteristics.input.avalanche_gain,
                pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
                common_voltage=valid_characteristics.input.common_voltage,
                gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
            ),
            adc_bit_resolution=adc_bit_resolution,
        )


@pytest.mark.parametrize(
    "adc_voltage_range",
    [
        pytest.param([1], id="adc_voltage_range too short"),
        pytest.param([1, 2, 3], id="adc_voltage_range too long"),
    ],
)
def test_invalid_initialization_adc_voltage_range(
    valid_characteristics: Params, adc_voltage_range
):
    with pytest.raises(ValueError, match="Voltage range must have length of 2"):
        _ = APDCharacteristics(
            roic_gain=0.5,
            bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
            avalanche_settings=AvalancheSettings(
                avalanche_gain=valid_characteristics.input.avalanche_gain,
                pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
                common_voltage=valid_characteristics.input.common_voltage,
                gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
            ),
            adc_voltage_range=adc_voltage_range,
        )


@pytest.mark.parametrize(
    "full_well_capacity",
    [
        pytest.param(-0.1, id="full_well_capacity too low"),
        pytest.param(10_000_001.0, id="full_well_capacity too high"),
    ],
)
def test_invalid_initialization_full_well_capacity(
    valid_characteristics: Params, full_well_capacity
):
    with pytest.raises(
        ValueError, match="'full_well_capacity' must be between 0 and 1e7"
    ):
        _ = APDCharacteristics(
            roic_gain=0.5,
            bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
            avalanche_settings=AvalancheSettings(
                avalanche_gain=valid_characteristics.input.avalanche_gain,
                pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
                common_voltage=valid_characteristics.input.common_voltage,
                gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
            ),
            full_well_capacity=full_well_capacity,
        )


def test_invalid_too_many_parameters_deprecated():
    with pytest.raises(ValueError, match="Please only specify two inputs"):
        _ = APDCharacteristics(
            roic_gain=0.5,
            bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
            avalanche_gain=1.0,
            pixel_reset_voltage=3.0,
            common_voltage=2.0,
        )


def test_invalid_too_many_parameters():
    with pytest.raises(ValueError, match="Too many parameters"):
        _ = APDCharacteristics(
            roic_gain=0.5,
            bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
            avalanche_settings=AvalancheSettings(
                avalanche_gain=1.0,
                pixel_reset_voltage=3.0,
                common_voltage=2.0,
                gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
            ),
        )


@pytest.mark.parametrize(
    "avalanche_gain,pixel_reset_voltage,common_voltage, exp_err",
    [
        pytest.param(
            None,
            None,
            None,
            r"Missing parameters\. Two of these parameters m",
            id="no parameters",
        ),
        pytest.param(
            1.0,
            None,
            None,
            r"\'avalanche_gain\' provided\. Missing",
            id="Only 'avalanche_gain'",
        ),
        pytest.param(
            None,
            None,
            2.0,
            r"\'avalanche_gain\' not provided and missing \'pixel_reset_voltage\'.",
            id="Only 'common_voltage'",
        ),
    ],
)
def test_invalid_not_enough_parameters(
    avalanche_gain, pixel_reset_voltage, common_voltage, exp_err
):
    with pytest.raises(ValueError, match=exp_err):
        _ = APDCharacteristics(
            roic_gain=0.5,
            bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
            avalanche_settings=AvalancheSettings(
                avalanche_gain=avalanche_gain,
                pixel_reset_voltage=pixel_reset_voltage,
                common_voltage=common_voltage,
                gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
            ),
        )


@pytest.mark.skip(reason="Method 'AvalancheSettings.__eq__' is not fully implemented")
def test_eq():
    obj1 = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=1.0,
            pixel_reset_voltage=2.0,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )
    obj2 = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=1.0,
            pixel_reset_voltage=2.0,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )
    obj3 = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=1.0,
            common_voltage=2.0,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )

    assert obj1 == obj2
    assert obj1 != obj3


@pytest.mark.parametrize(
    "quantum_efficiency, exp_quantum_efficiency",
    [
        pytest.param(0.0, 0.0, id="QE == 0"),
        pytest.param(1.0, 1.0, id="QE == 1"),
        pytest.param(
            -0.1,
            pytest.raises(
                ValueError,
                match="'quantum_efficiency' values must be between 0\\.0 and 1\\.0",
            ),
            id="QE < 0",
        ),
        pytest.param(
            1.1,
            pytest.raises(
                ValueError,
                match="'quantum_efficiency' values must be between 0\\.0 and 1\\.0",
            ),
            id="QE > 1",
        ),
    ],
)
def test_valid_qe(
    valid_characteristics: Params, quantum_efficiency, exp_quantum_efficiency
):
    obj = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )

    if isinstance(exp_quantum_efficiency, AbstractContextManager):
        with exp_quantum_efficiency:
            obj.quantum_efficiency = quantum_efficiency
    else:
        obj.quantum_efficiency = quantum_efficiency
        assert obj.quantum_efficiency == exp_quantum_efficiency


@pytest.mark.parametrize(
    "avalanche_gain, exp_avalanche_gain",
    [
        pytest.param(1.0, 1.0, id="avalanche_gain == 1."),
        pytest.param(1000.0, 1000.0, id="avalanche_gain == 1000."),
        pytest.param(
            0.9,
            pytest.raises(ValueError, match=r"Invalid \'avalanche_gain"),
            id="avalanche_gain < 1",
        ),
        pytest.param(
            1000.1,
            pytest.raises(ValueError, match=r"Invalid \'avalanche_gain"),
            id="avalanche_gain > 1000",
        ),
    ],
)
def test_valid_avalanche_gain(
    valid_characteristics: Params,
    avalanche_gain,
    exp_avalanche_gain,
):
    obj: APDCharacteristics = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )

    if isinstance(exp_avalanche_gain, AbstractContextManager):
        with exp_avalanche_gain:
            obj.avalanche_settings.avalanche_gain = avalanche_gain
    else:
        obj.avalanche_settings.avalanche_gain = avalanche_gain
        assert obj.avalanche_settings.avalanche_gain == exp_avalanche_gain


@pytest.mark.parametrize(
    "pixel_reset_voltage, exp_pixel_reset_voltage",
    [
        pytest.param(1.0, 1.0, id="pixel_reset_voltage == 1."),
        pytest.param(1000.0, 1000.0, id="pixel_reset_voltage == 1000."),
    ],
)
def test_valid_pixel_reset_voltage(
    valid_characteristics: Params,
    pixel_reset_voltage,
    exp_pixel_reset_voltage,
):
    obj = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )

    if isinstance(exp_pixel_reset_voltage, AbstractContextManager):
        with exp_pixel_reset_voltage:
            obj.avalanche_settings.pixel_reset_voltage = pixel_reset_voltage
    else:
        obj.avalanche_settings.pixel_reset_voltage = pixel_reset_voltage
        assert obj.avalanche_settings.pixel_reset_voltage == exp_pixel_reset_voltage


@pytest.mark.parametrize(
    "adc_bit_resolution, exp_adc_bit_resolution",
    [
        pytest.param(4, 4, id="adc_bit_resolution == 4"),
        pytest.param(64, 64, id="adc_bit_resolution == 64"),
        pytest.param(
            3,
            pytest.raises(
                ValueError, match="'adc_bit_resolution' must be between 4 and 64"
            ),
            id="adc_bit_resolution < 4",
        ),
        pytest.param(
            65,
            pytest.raises(
                ValueError, match="'adc_bit_resolution' must be between 4 and 64"
            ),
            id="adc_bit_resolution > 64",
        ),
    ],
)
def test_valid_adc_bit_resolution(
    valid_characteristics: Params,
    adc_bit_resolution,
    exp_adc_bit_resolution,
):
    obj = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )

    if isinstance(exp_adc_bit_resolution, AbstractContextManager):
        with exp_adc_bit_resolution:
            obj.adc_bit_resolution = adc_bit_resolution
    else:
        obj.adc_bit_resolution = adc_bit_resolution
        assert obj.adc_bit_resolution == exp_adc_bit_resolution


@pytest.mark.parametrize(
    "full_well_capacity, exp_full_well_capacity",
    [
        pytest.param(0, 0, id="full_well_capacity == 0"),
        pytest.param(10_000_000, 10_000_000, id="full_well_capacity == 10_000_000"),
        pytest.param(
            -1,
            pytest.raises(ValueError, match="'full_well_capacity' must be between"),
            id="full_well_capacity < 0",
        ),
        pytest.param(
            10_000_001,
            pytest.raises(ValueError, match="'full_well_capacity' must be between"),
            id="full_well_capacity < 10_000_001",
        ),
    ],
)
def test_valid_full_well_capacity(
    valid_characteristics: Params,
    full_well_capacity,
    exp_full_well_capacity,
):
    obj = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )

    if isinstance(exp_full_well_capacity, AbstractContextManager):
        with exp_full_well_capacity:
            obj.full_well_capacity = full_well_capacity
    else:
        obj.full_well_capacity = full_well_capacity
        assert obj.full_well_capacity == exp_full_well_capacity


@pytest.mark.parametrize(
    "adc_voltage_range, exp_adc_voltage_range",
    [
        ((2.0, 3.0), (2.0, 3.0)),
        ((3.0, 2.0), (3.0, 2.0)),
    ],
)
def test_valid_adc_voltage_range(
    valid_characteristics: Params,
    adc_voltage_range,
    exp_adc_voltage_range,
):
    obj = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )

    if isinstance(exp_adc_voltage_range, AbstractContextManager):
        with exp_adc_voltage_range:
            obj.adc_voltage_range = adc_voltage_range
    else:
        obj.adc_voltage_range = adc_voltage_range
        assert obj.adc_voltage_range == exp_adc_voltage_range


@pytest.mark.parametrize(
    "common_voltage, exp_common_voltage",
    [
        pytest.param(1.0, 1.0, id="common_voltage == 1."),
        pytest.param(1000.0, 1000.0, id="common_voltage == 1000."),
    ],
)
def test_valid_common_voltage(
    valid_characteristics: Params,
    common_voltage,
    exp_common_voltage,
):
    obj = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )

    if isinstance(exp_common_voltage, AbstractContextManager):
        with exp_common_voltage:
            obj.avalanche_settings.common_voltage = common_voltage
    else:
        obj.common_voltage = common_voltage
        assert obj.avalanche_settings.common_voltage == exp_common_voltage


@pytest.mark.parametrize(
    "characteristics",
    [
        APDCharacteristics.build(
            {
                "roic_gain": 0.5,
                "bias_to_node": {"values": [(2.65, 73.7)]},
                "avalanche_settings": {
                    "avalanche_gain": 1.0,
                    "pixel_reset_voltage": 2.65,
                    "gain_to_bias": {"values": [(1.0, 2.65)]},
                    "bias_to_gain": {"values": [(2.65, 1.0)]},
                },
            }
        ),
        APDCharacteristics.build(
            {
                "roic_gain": 0.5,
                "bias_to_node": {"values": [(4.0, 60.0)]},
                "avalanche_settings": {
                    "avalanche_gain": 10.0,
                    "pixel_reset_voltage": 4.0,
                    "gain_to_bias": {"values": [(10.0, 4.0)]},
                    "bias_to_gain": {"values": [(4.0, 10.0)]},
                },
            }
        ),
        APDCharacteristics.build(
            {
                "roic_gain": 0.5,
                "bias_to_node": {"values": [(1.0, 100.0)]},
                "avalanche_settings": {
                    "pixel_reset_voltage": 3.0,
                    "common_voltage": 2.0,
                    "gain_to_bias": {"values": [(1.0, 1.0)]},
                    "bias_to_gain": {"values": [(1.0, 1.0)]},
                },
            }
        ),
    ],
)
def test_charge_to_volt_conversion(characteristics):
    assert isinstance(characteristics, APDCharacteristics)

    capacitance_farads = characteristics.bias_to_node_capacitance(
        characteristics.avalanche_settings.avalanche_bias
    )
    expected = characteristics.roic_gain * (const.e.value / capacitance_farads)

    assert characteristics.charge_to_volt_conversion == pytest.approx(expected)


@pytest.mark.parametrize(
    "characteristics",
    [
        APDCharacteristics.build(
            {
                "roic_gain": 0.5,
                "bias_to_node": {"values": [(2.65, 73.7), (4.0, 60.0)]},
                "avalanche_settings": {
                    "avalanche_gain": 1.0,
                    "pixel_reset_voltage": 2.65,
                    "gain_to_bias": {"values": [(1.0, 2.65), (10.0, 4.0)]},
                    "bias_to_gain": {"values": [(2.65, 1.0), (4.0, 10.0)]},
                },
                "quantum_efficiency": 0.8,
                "adc_bit_resolution": 14,
                "adc_voltage_range": (0.0, 16.0),
            }
        ),
        APDCharacteristics.build(
            {
                "roic_gain": 0.5,
                "bias_to_node": {"values": [(1.0, 73.7)]},
                "avalanche_settings": {
                    "avalanche_gain": 1.0,
                    "common_voltage": 2.0,
                    "gain_to_bias": {"values": [(1.0, 1.0)]},
                    "bias_to_gain": {"values": [(1.0, 1.0)]},
                },
                "quantum_efficiency": 0.9,
                "adc_bit_resolution": 16,
                "adc_voltage_range": (0.0, 16.0),
            }
        ),
        APDCharacteristics.build(
            {
                "roic_gain": 0.5,
                "bias_to_node": {
                    "values": [(1.0, 3.0)]  # very small capacitance → high gain
                },
                "avalanche_settings": {
                    "pixel_reset_voltage": 3.0,
                    "common_voltage": 2.0,
                    "gain_to_bias": {"values": [(1.0, 1.0)]},
                    "bias_to_gain": {"values": [(1.0, 1.0)]},
                },
                "quantum_efficiency": 0.7,
                "adc_bit_resolution": 32,
                "adc_voltage_range": (0.0, 16.0),
            }
        ),
    ],
)
def test_system_gain(characteristics):
    assert isinstance(characteristics, APDCharacteristics)

    qe = characteristics.quantum_efficiency
    gain = characteristics.avalanche_settings.avalanche_gain
    c2v = characteristics.charge_to_volt_conversion
    bits = characteristics.adc_bit_resolution
    vmin, vmax = characteristics.adc_voltage_range

    expected = pytest.approx((qe * gain * c2v * 2**bits) / (vmax - vmin))

    assert characteristics.system_gain == expected


@pytest.mark.parametrize(
    "characteristics",
    [
        APDCharacteristics.build(
            {
                "roic_gain": 0.5,
                "bias_to_node": {"values": [(2.65, 73.7), (4.0, 60.0)]},
                "avalanche_settings": {
                    "avalanche_gain": 1.0,
                    "pixel_reset_voltage": 2.0,
                    "gain_to_bias": {"values": [(1.0, 2.65), (10.0, 4.0)]},
                    "bias_to_gain": {"values": [(2.65, 1.0), (4.0, 10.0)]},
                },
            }
        ),
        APDCharacteristics.build(
            {
                "roic_gain": 0.5,
                "bias_to_node": {"values": [(1.0, 73.7)]},
                "avalanche_settings": {
                    "avalanche_gain": 1.0,
                    "common_voltage": 2.0,
                    "gain_to_bias": {"values": [(1.0, 1.0)]},
                    "bias_to_gain": {"values": [(1.0, 1.0)]},
                },
                "quantum_efficiency": 0.9,
            }
        ),
        APDCharacteristics.build(
            {
                "roic_gain": 0.5,
                "bias_to_node": {"values": [(1.0, 3.0)]},
                "avalanche_settings": {
                    "common_voltage": 2.0,
                    "pixel_reset_voltage": 3.0,
                    "gain_to_bias": {"values": [(1.0, 1.0)]},
                    "bias_to_gain": {"values": [(1.0, 1.0)]},
                },
                "quantum_efficiency": 0.7,
                "adc_bit_resolution": 32,
                "adc_voltage_range": (0.0, 16.0),
            }
        ),
    ],
)
def test_numbytes(characteristics):
    assert isinstance(characteristics, APDCharacteristics)
    assert characteristics.numbytes > 1000


@pytest.mark.parametrize(
    "bias, expected",
    [
        (3.0, pytest.approx(60.0e-15)),
        (2.0, pytest.approx(90.0e-15)),
    ],
)
def test_bias_to_node_capacitance_valid(bias, expected):
    apd = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
        avalanche_settings=AvalancheSettings(
            pixel_reset_voltage=3.0,
            common_voltage=0.0,
            gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
            bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
        ),
    )

    result = apd.bias_to_node_capacitance(bias)
    assert result == expected


@pytest.mark.parametrize(
    "characteristics, exp_dct",
    [
        (
            APDCharacteristics(
                roic_gain=0.5,
                bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
                avalanche_settings=AvalancheSettings(
                    avalanche_gain=1.0,
                    pixel_reset_voltage=2.0,
                    gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                    bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
                ),
            ),
            {
                "roic_gain": 0.5,
                "bias_to_node": {"values": [(2.65, 73.7), (4.0, 60.0)]},
                "avalanche_settings": {
                    "avalanche_gain": 1.0,
                    "common_voltage": None,
                    "pixel_reset_voltage": 2.0,
                    # "gain_to_bias": {"function": None},
                    "bias_to_gain": {"values": [(2.65, 1.0), (4.0, 10.0)]},
                },
                "adc_bit_resolution": None,
                "adc_voltage_range": None,
                "full_well_capacity": None,
                "quantum_efficiency": None,
            },
        ),
        (
            APDCharacteristics(
                roic_gain=0.5,
                bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
                avalanche_settings=AvalancheSettings(
                    avalanche_gain=1.0,
                    common_voltage=2.0,
                    gain_to_bias=ConverterFunction(lambda gain: 0.15 * gain + 2.5),
                    bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
                ),
                quantum_efficiency=0.9,
            ),
            {
                "roic_gain": 0.5,
                "bias_to_node": {"values": [(2.65, 73.7), (4.0, 60.0)]},
                "avalanche_settings": {
                    "avalanche_gain": 1.0,
                    "common_voltage": 2.0,
                    "pixel_reset_voltage": None,  # was 3.0
                    "bias_to_gain": {"values": [(2.65, 1.0), (4.0, 10.0)]},
                    # "gain_to_bias": {"function": None},
                },
                "adc_bit_resolution": None,
                "adc_voltage_range": None,
                "full_well_capacity": None,
                "quantum_efficiency": 0.9,
            },
        ),
    ],
)
def test_to_dict_from_dict(characteristics, exp_dct):
    # Check '.to_dict'
    dct = characteristics.to_dict()
    dct_copied = deepcopy(dct)

    # Remove 'gain_to_bias'
    dct_gain_to_bias = dct["avalanche_settings"].pop("gain_to_bias")

    assert dct == exp_dct
    assert list(dct_gain_to_bias) == ["function"]
    assert isinstance(dct_gain_to_bias["function"], bytes)

    # Check '.from_dict'
    new_characteristics = APDCharacteristics.from_dict(dct_copied)
    assert isinstance(new_characteristics, APDCharacteristics)

    new_dct = new_characteristics.to_dict()
    assert new_dct == dct_copied
    # assert APDCharacteristics.from_dict(dct).to_dict() == exp_dct


def test_bias_to_node_func_callable():
    apd = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterFunction(lambda b: 42 - b),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=5.0,
            pixel_reset_voltage=2.5,
            gain_to_bias=ConverterFunction(lambda g: 0.2 * g + 1.0),
            bias_to_gain=ConverterFunction(lambda b: (b - 1.0) / 0.2),
        ),
    )
    result = apd.bias_to_node_capacitance(2.0)
    assert result == pytest.approx(40.0e-15)


def test_bias_to_node_func_list():
    from collections.abc import Callable

    func_data = [(1.0, 40.0), (2.0, 38.0), (3.0, 36.0)]
    apd = APDCharacteristics(
        roic_gain=0.5,
        bias_to_node=ConverterValues(func_data),
        avalanche_settings=AvalancheSettings(
            avalanche_gain=3.0,
            pixel_reset_voltage=3.5,
            gain_to_bias=ConverterFunction(lambda g: g - 1),
            bias_to_gain=ConverterFunction(lambda b: b + 1),
        ),
    )

    assert isinstance(apd.bias_to_node_capacitance, Callable)
    assert apd.bias_to_node_capacitance(2.0) == pytest.approx(38.0e-15)


# def test_bias_to_node_func_csv():
#     import csv
#     import os
#     import tempfile
#     from collections.abc import Callable
#
#     with tempfile.NamedTemporaryFile(
#         mode="w", delete=False, newline="", suffix=".csv"
#     ) as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["bias", "capacitance"])
#         writer.writerow([1.0, 40.0])
#         writer.writerow([2.0, 38.0])
#         writer.writerow([3.0, 36.0])
#         csv_path = csvfile.name
#
#     try:
#         apd = APDCharacteristics(
#             roic_gain=0.5,
#             avalanche_gain=3.0,
#             pixel_reset_voltage=3.5,
#             bias_to_node_func=csv_path,
#             bias_to_gain_func=lambda b: b + 1,
#         )
#
#         assert isinstance(apd.bias_to_node_capacitance, Callable)
#         assert apd.bias_to_node_capacitance(2.0) == pytest.approx(38.0e-15)
#
#     finally:
#         os.remove(csv_path)
#
#
# def test_gain_to_bias_func_list():
#     from collections.abc import Callable
#
#     func_data = [(1.0, 2.5), (2.0, 3.0), (3.0, 3.5)]
#
#     apd = APDCharacteristics(
#         roic_gain=0.5,
#         avalanche_gain=2.0,
#         pixel_reset_voltage=5.5,
#         gain_to_bias_func=func_data,
#         bias_to_node_func=lambda b: 40.0 - b,
#     )
#
#     assert isinstance(apd.gain_to_bias, Callable)
#     assert apd.gain_to_bias(2.0) == pytest.approx(3.0)
#
#
# def test_gain_to_bias_inverted_from_bias_to_gain():
#     from collections.abc import Callable
#
#     bias_to_gain = lambda b: b + 1
#
#     apd = APDCharacteristics(
#         roic_gain=0.5,
#         avalanche_gain=3.0,
#         pixel_reset_voltage=5.5,
#         bias_to_gain_func=bias_to_gain,
#         bias_to_node_func=lambda b: 40.0 - b,
#     )
#
#     gain_to_bias_func = apd.gain_to_bias
#
#     assert isinstance(gain_to_bias_func, Callable)
#     assert gain_to_bias_func(3.0) == pytest.approx(2.0)
#
#
# def test_bias_to_gain_inverted_from_gain_to_bias():
#     from collections.abc import Callable
#
#     # Exact inverse: gain = bias + 1
#     gain_to_bias = lambda g: g - 1
#
#     apd = APDCharacteristics(
#         roic_gain=0.5,
#         avalanche_gain=3.0,
#         pixel_reset_voltage=4.0,
#         gain_to_bias_func=gain_to_bias,
#         bias_to_node_func=lambda b: 40.0 - b,
#     )
#
#     bias_to_gain_func = apd.bias_to_gain
#
#     # Check that the inversion is callable and returns expected value
#     assert isinstance(bias_to_gain_func, Callable)
#     assert bias_to_gain_func(2.0) == pytest.approx(3.0)
#
#
# def test_invalid_bias_to_gain_func_range():
#     # This function returns values outside the valid gain range [1.0, 1000.0]
#     def bad_bias_to_gain(bias: float) -> float:
#         if bias < 1.0:
#             return 0.5  # Too low
#         elif bias > 2.0:
#             return 2000.0  # Too high
#         return 10.0  # Valid in the middle
#
#     with pytest.raises(ValueError, match="'apd_gain' must be between 1.0 and 1000.0"):
#         APDCharacteristics(
#             roic_gain=0.5,
#             avalanche_gain=2000.0,  # Will trigger validation after func call
#             pixel_reset_voltage=3.5,
#             bias_to_gain_func=bad_bias_to_gain,
#             bias_to_node_func=lambda b: 30.0,  # Arbitrary valid function
#         )
#
#
# def test_gain_to_bias_inversion_from_bias_to_gain_func():
#     # Simple monotonic function
#     bias_to_gain = lambda b: 2 * b  # Gain = 2 * Bias -> Bias = Gain / 2
#
#     apd = APDCharacteristics(
#         roic_gain=0.5,
#         avalanche_gain=6.0,
#         pixel_reset_voltage=3.0,
#         bias_to_gain_func=bias_to_gain,
#         bias_to_node_func=lambda b: 40.0,
#     )
#
#     gain_to_bias_func = apd.gain_to_bias
#     assert callable(gain_to_bias_func)
#     assert gain_to_bias_func(6.0) == pytest.approx(3.0)
#
#
# def test_inversion_failure():
#     # Constant function (not invertible)
#     bias_to_gain = lambda b: 5.0
#
#     apd = APDCharacteristics(
#         roic_gain=0.5,
#         pixel_reset_voltage=3.5,
#         common_voltage=0.5,
#         bias_to_gain_func=bias_to_gain,
#         bias_to_node_func=lambda b: 40.0,
#     )
#
#     gain_to_bias_func = apd.gain_to_bias
#
#     with pytest.raises(ValueError, match="Cannot invert function"):
#         gain_to_bias_func(6.0)
#
#
# def test_gain_to_bias_from_csv(tmp_path):
#     import pandas as pd
#
#     df = pd.DataFrame({"gain": [1.0, 2.0, 3.0], "bias": [2.0, 3.0, 4.0]})
#     csv_path = tmp_path / "gain_bias.csv"
#     df.to_csv(csv_path, index=False)
#
#     apd = APDCharacteristics(
#         roic_gain=0.5,
#         avalanche_gain=2.0,
#         pixel_reset_voltage=5.0,
#         gain_to_bias_func=str(csv_path),
#         bias_to_node_func=lambda b: 30.0,
#     )
#
#     func = apd.gain_to_bias
#     assert callable(func)
#     assert func(2.0) == pytest.approx(3.0)
#
#
# def test_invalid_gain_to_bias_input_type():
#     with pytest.raises(TypeError, match="Function input must be callable"):
#         APDCharacteristics(
#             roic_gain=0.5,
#             avalanche_gain=5.0,
#             pixel_reset_voltage=3.0,
#             gain_to_bias_func=1234,  # Invalid
#             bias_to_node_func=lambda b: 30.0,
#         )
#
#
# class DummyGeometry:
#     def __init__(self, shape, channel_map):
#         from types import SimpleNamespace
#
#         self.shape = shape
#         self.channels = SimpleNamespace(readout_position=channel_map)
#         self._channel_coords = {
#             "A": (slice(0, 2), slice(0, 2)),
#             "B": (slice(0, 2), slice(2, 4)),
#         }
#
#     def get_channel_coord(self, channel):
#         return self._channel_coords[channel]
#
#
# def test_per_channel_charge_to_volt_conversion():
#     import numpy as np
#
#     # Create a dictionary with gain per channel
#     channel_gains = {
#         "A": 0.9,
#         "B": 0.5,
#     }
#
#     # Create a dummy detector
#     apd = APDCharacteristics(
#         roic_gain=0.5,
#         avalanche_gain=3.0,
#         pixel_reset_voltage=3.5,
#         bias_to_gain_func=lambda b: b + 1,
#         bias_to_node_func=lambda b: 30.0,
#     )
#
#     # Override charge_to_volt_conversion with channel dict
#     apd._charge_to_volt_conversion = channel_gains
#
#     # Create and assign dummy geometry (4x4 pixels split in two 2x2 regions)
#     geometry = DummyGeometry(shape=(2, 4), channel_map=channel_gains)
#     apd.initialize(geometry)
#
#     # Check that the 2D gain map is correctly built
#     assert apd._channels_gain.shape == (2, 4)
#     assert np.all(apd._channels_gain[:, 0:2] == 0.9)  # Channel A
#     assert np.all(apd._channels_gain[:, 2:4] == 0.5)  # Channel B
#
#
# def test_charge_to_volt_conversion_new_auto():
#     apd = APDCharacteristics(
#         roic_gain=0.5,
#         avalanche_gain=3.0,
#         pixel_reset_voltage=3.5,
#         bias_to_node_func=lambda b: 40,
#         bias_to_gain_func=lambda b: b + 1,
#     )
#
#     expected = apd.detector_gain(capacitance=40e-15, roic_gain=0.5)
#     assert apd.charge_to_volt_conversion == pytest.approx(expected, rel=1e-6)
#
#
# def test_system_gain_new():
#     apd = APDCharacteristics(
#         roic_gain=0.5,
#         quantum_efficiency=0.8,
#         avalanche_gain=2.0,
#         pixel_reset_voltage=3.5,
#         adc_bit_resolution=12,
#         adc_voltage_range=(0.0, 3.3),
#         bias_to_node_func=lambda b: 40,
#         bias_to_gain_func=lambda b: b + 1,
#     )
#
#     expected_ctov = apd.detector_gain(40e-15, 0.5)
#     expected_gain = 0.8 * 2.0 * expected_ctov * (2**12) / (3.3 - 0.0)
#     assert apd.system_gain == pytest.approx(expected_gain)
#
#
# def test_initialize_with_global_gain():
#     import numpy as np
#
#     class MockGeometry:
#         shape = (2, 2)
#
#         class Channels:
#             readout_position = {"A": (0, 0), "B": (0, 1)}
#
#         channels = Channels()
#
#         def get_channel_coord(self, ch):
#             return ((0, 1), (0, 1)) if ch == "A" else ((1, 2), (1, 2))
#
#     apd = APDCharacteristics(
#         roic_gain=1e-6,
#         avalanche_gain=2.0,
#         pixel_reset_voltage=3.5,
#         bias_to_node_func=lambda b: 1e-10,  # large capacitance
#         bias_to_gain_func=lambda b: b + 1,
#     )
#
#     apd.initialize(MockGeometry())
#
#     # Check scalar gain
#     assert isinstance(apd._channels_gain, float | np.floating)
#     assert apd._channels_gain < 100.0
#
#
# def test_initialize_with_per_channel_gains():
#     import numpy as np
#
#     class MockGeometry:
#         shape = (2, 2)
#
#         class Channels:
#             readout_position = {"A": (0, 0), "B": (0, 1)}
#
#         channels = Channels()
#
#         def get_channel_coord(self, ch):
#             return (
#                 (slice(0, 1), slice(0, 1)) if ch == "A" else (slice(1, 2), slice(0, 1))
#             )
#
#     apd = APDCharacteristics(
#         roic_gain=1e-6,
#         avalanche_gain=2.0,
#         pixel_reset_voltage=3.5,
#         bias_to_node_func=lambda b: 1e-10,
#         bias_to_gain_func=lambda b: b + 1,
#     )
#
#     # Manually call _build_channels_gain with a dict (simulates per-channel override)
#     apd._geometry = MockGeometry()
#     apd._build_channels_gain({"A": 0.5, "B": 0.8})
#
#     assert isinstance(apd._channels_gain, np.ndarray)
#     assert apd._channels_gain.shape == (2, 2)
#     assert apd._channels_gain[0, 0] == 0.5  # Channel A
#     assert apd._channels_gain[1, 0] == 0.8  # Channel B
