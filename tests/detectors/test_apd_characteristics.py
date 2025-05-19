#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from contextlib import AbstractContextManager
from dataclasses import dataclass

import pytest

from pyxel.detectors import APDCharacteristics


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
def test_valid_initialization(
    valid_characteristics: Params,
    quantum_efficiency,
    adc_bit_resolution,
    adc_voltage_range,
    full_well_capacity,
):
    obj = APDCharacteristics(
        roic_gain=0.5,
        avalanche_gain=valid_characteristics.input.avalanche_gain,
        pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
        common_voltage=valid_characteristics.input.common_voltage,
        quantum_efficiency=quantum_efficiency,
        adc_bit_resolution=adc_bit_resolution,
        adc_voltage_range=adc_voltage_range,
        full_well_capacity=full_well_capacity,
    )

    assert obj.avalanche_gain == pytest.approx(
        valid_characteristics.output.avalanche_gain
    )
    assert obj.pixel_reset_voltage == pytest.approx(
        valid_characteristics.output.pixel_reset_voltage
    )
    assert obj.common_voltage == pytest.approx(
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
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
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
    with pytest.raises(
        ValueError, match="'apd_gain' must be between 1\\.0 and 1000\\.0"
    ):
        _ = APDCharacteristics(
            roic_gain=0.5,
            avalanche_gain=avalanche_gain,
            pixel_reset_voltage=valid_characteristics_only_avalanche.input.pixel_reset_voltage,
            common_voltage=valid_characteristics_only_avalanche.input.common_voltage,
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
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
            adc_bit_resolution=adc_bit_resolution,
        )


@pytest.mark.parametrize(
    "adc_voltage_range",
    [
        pytest.param(
            [
                1,
            ],
            id="adc_voltage_range too short",
        ),
        pytest.param([1, 2, 3], id="adc_voltage_range too long"),
    ],
)
def test_invalid_initialization_adc_voltage_range(
    valid_characteristics: Params, adc_voltage_range
):
    with pytest.raises(ValueError, match="Voltage range must have length of 2"):
        _ = APDCharacteristics(
            roic_gain=0.5,
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
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
            avalanche_gain=valid_characteristics.input.avalanche_gain,
            pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
            common_voltage=valid_characteristics.input.common_voltage,
            full_well_capacity=full_well_capacity,
        )


def test_invalid_too_many_parameters():
    with pytest.raises(ValueError, match="Please only specify two inputs"):
        _ = APDCharacteristics(
            roic_gain=0.5,
            avalanche_gain=1.0,
            pixel_reset_voltage=3.0,
            common_voltage=2.0,
        )


@pytest.mark.parametrize(
    "avalanche_gain,pixel_reset_voltage,common_voltage, exp_err",
    [
        pytest.param(
            None, None, None, "Not enough input parameters", id="no parameters"
        ),
        pytest.param(
            1.0,
            None,
            None,
            "Only 'avalanche_gain', missing parameter 'pixel_reset_voltage' or 'common_voltage'",
            id="Only 'avalanche_gain'",
        ),
        pytest.param(
            None,
            None,
            2.0,
            "Only 'common_voltage', missing parameter 'pixel_reset_voltage' or 'avalanche_gain'",
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
            avalanche_gain=avalanche_gain,
            pixel_reset_voltage=pixel_reset_voltage,
            common_voltage=common_voltage,
        )


def test_eq():
    obj1 = APDCharacteristics(
        roic_gain=0.5, avalanche_gain=1.0, pixel_reset_voltage=2.0
    )
    obj2 = APDCharacteristics(
        roic_gain=0.5, avalanche_gain=1.0, pixel_reset_voltage=2.0
    )
    obj3 = APDCharacteristics(roic_gain=0.5, avalanche_gain=1.0, common_voltage=2.0)

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
        avalanche_gain=valid_characteristics.input.avalanche_gain,
        pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
        common_voltage=valid_characteristics.input.common_voltage,
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
            pytest.raises(
                ValueError, match=r"\'apd_gain\' values must be between 1\.0 and 1000\."
            ),
            id="avalanche_gain < 1",
        ),
        pytest.param(
            1000.1,
            pytest.raises(
                ValueError, match=r"\'apd_gain\' values must be between 1\.0 and 1000\."
            ),
            id="avalanche_gain > 1000",
        ),
    ],
)
def test_valid_avalanche_gain(
    valid_characteristics: Params,
    avalanche_gain,
    exp_avalanche_gain,
):
    obj = APDCharacteristics(
        roic_gain=0.5,
        avalanche_gain=valid_characteristics.input.avalanche_gain,
        pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
        common_voltage=valid_characteristics.input.common_voltage,
    )

    if isinstance(exp_avalanche_gain, AbstractContextManager):
        with exp_avalanche_gain:
            obj.avalanche_gain = avalanche_gain
    else:
        obj.avalanche_gain = avalanche_gain
        assert obj.avalanche_gain == exp_avalanche_gain


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
        avalanche_gain=valid_characteristics.input.avalanche_gain,
        pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
        common_voltage=valid_characteristics.input.common_voltage,
    )

    if isinstance(exp_pixel_reset_voltage, AbstractContextManager):
        with exp_pixel_reset_voltage:
            obj.pixel_reset_voltage = pixel_reset_voltage
    else:
        obj.pixel_reset_voltage = pixel_reset_voltage
        assert obj.pixel_reset_voltage == exp_pixel_reset_voltage


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
        avalanche_gain=valid_characteristics.input.avalanche_gain,
        pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
        common_voltage=valid_characteristics.input.common_voltage,
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
        avalanche_gain=valid_characteristics.input.avalanche_gain,
        pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
        common_voltage=valid_characteristics.input.common_voltage,
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
        avalanche_gain=valid_characteristics.input.avalanche_gain,
        pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
        common_voltage=valid_characteristics.input.common_voltage,
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
        avalanche_gain=valid_characteristics.input.avalanche_gain,
        pixel_reset_voltage=valid_characteristics.input.pixel_reset_voltage,
        common_voltage=valid_characteristics.input.common_voltage,
    )

    if isinstance(exp_common_voltage, AbstractContextManager):
        with exp_common_voltage:
            obj.common_voltage = common_voltage
    else:
        obj.common_voltage = common_voltage
        assert obj.common_voltage == exp_common_voltage


@pytest.mark.parametrize(
    "characteristics, exp_charge_to_volt",
    [
        (
            APDCharacteristics(
                roic_gain=0.5, avalanche_gain=1.0, pixel_reset_voltage=2.0
            ),
            2.1695011970209884e-06,
        ),
        (
            APDCharacteristics(roic_gain=0.5, avalanche_gain=1.0, common_voltage=2.0),
            2.1695011970209884e-06,
        ),
        (
            APDCharacteristics(
                roic_gain=0.5, common_voltage=2.0, pixel_reset_voltage=3.0
            ),
            1.7227705741935481e-06,
        ),
    ],
)
def test_charge_to_volt_conversion(characteristics, exp_charge_to_volt):
    assert isinstance(characteristics, APDCharacteristics)
    assert characteristics.charge_to_volt_conversion == pytest.approx(
        exp_charge_to_volt
    )


@pytest.mark.parametrize(
    "characteristics, exp_system_gain",
    [
        (
            APDCharacteristics(
                roic_gain=0.5,
                avalanche_gain=1.0,
                pixel_reset_voltage=2.0,
                quantum_efficiency=0.8,
                adc_bit_resolution=14,
                adc_voltage_range=(0.0, 16.0),
            ),
            0.0017772553805995937,
        ),
        (
            APDCharacteristics(
                roic_gain=0.5,
                avalanche_gain=1.0,
                common_voltage=2.0,
                quantum_efficiency=0.9,
                adc_bit_resolution=16,
                adc_voltage_range=(0.0, 16.0),
            ),
            0.007997649212698172,
        ),
        (
            APDCharacteristics(
                roic_gain=0.5,
                common_voltage=2.0,
                pixel_reset_voltage=3.0,
                quantum_efficiency=0.7,
                adc_bit_resolution=32,
                adc_voltage_range=(0.0, 16.0),
            ),
            323.7168932669188,
        ),
    ],
)
def test_system_gain(characteristics, exp_system_gain):
    assert isinstance(characteristics, APDCharacteristics)
    assert characteristics.system_gain == pytest.approx(exp_system_gain)


@pytest.mark.parametrize(
    "characteristics",
    [
        APDCharacteristics(
            roic_gain=0.5,
            avalanche_gain=1.0,
            pixel_reset_voltage=2.0,
        ),
        APDCharacteristics(
            roic_gain=0.5,
            avalanche_gain=1.0,
            common_voltage=2.0,
            quantum_efficiency=0.9,
        ),
        APDCharacteristics(
            roic_gain=0.5,
            common_voltage=2.0,
            pixel_reset_voltage=3.0,
            quantum_efficiency=0.7,
            adc_bit_resolution=32,
            adc_voltage_range=(0.0, 16.0),
        ),
    ],
)
def test_numbytes(characteristics):
    assert isinstance(characteristics, APDCharacteristics)
    assert characteristics.numbytes > 1000


@pytest.mark.parametrize(
    "value, exp_result",
    [
        (3.0, 3.605e-14),
        (
            0.9,
            pytest.raises(
                ValueError, match="Node capacitance calculation is inaccurate"
            ),
        ),
    ],
)
def test_bias_to_node_capacitance_saphira(value, exp_result):
    if isinstance(exp_result, AbstractContextManager):
        with exp_result:
            _ = APDCharacteristics.bias_to_node_capacitance_saphira(bias=value)

    else:
        result = APDCharacteristics.bias_to_node_capacitance_saphira(bias=value)
        assert result == pytest.approx(exp_result)


@pytest.mark.parametrize(
    "characteristics, exp_dct",
    [
        (
            APDCharacteristics(
                roic_gain=0.5,
                avalanche_gain=1.0,
                pixel_reset_voltage=2.0,
            ),
            {
                "adc_bit_resolution": None,
                "adc_voltage_range": None,
                "avalanche_gain": 1.0,
                "common_voltage": None,
                "full_well_capacity": None,
                "pixel_reset_voltage": 2.0,
                "quantum_efficiency": None,
                "roic_gain": 0.5,
            },
        ),
        (
            APDCharacteristics(
                roic_gain=0.5,
                avalanche_gain=1.0,
                common_voltage=2.0,
                quantum_efficiency=0.9,
            ),
            {
                "adc_bit_resolution": None,
                "adc_voltage_range": None,
                "avalanche_gain": 1.0,
                "common_voltage": 2.0,
                "full_well_capacity": None,
                "pixel_reset_voltage": None,
                "quantum_efficiency": 0.9,
                "roic_gain": 0.5,
            },
        ),
        (
            APDCharacteristics(
                roic_gain=0.5,
                common_voltage=2.0,
                pixel_reset_voltage=3.0,
                quantum_efficiency=0.7,
                adc_bit_resolution=32,
                adc_voltage_range=(0.0, 16.0),
            ),
            {
                "adc_bit_resolution": 32,
                "adc_voltage_range": (0.0, 16.0),
                "avalanche_gain": None,
                "common_voltage": 2.0,
                "full_well_capacity": None,
                "pixel_reset_voltage": 3.0,
                "quantum_efficiency": 0.7,
                "roic_gain": 0.5,
            },
        ),
    ],
)
def test_to_dict_from_dict(characteristics, exp_dct):
    assert isinstance(characteristics, APDCharacteristics)

    dct = characteristics.to_dict()
    assert dct == exp_dct

    obj = APDCharacteristics.from_dict(exp_dct)
    assert obj == characteristics
