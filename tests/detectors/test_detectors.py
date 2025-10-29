#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import copy
from pathlib import Path

import pytest

from pyxel.detectors import (
    APD,
    CCD,
    CMOS,
    MKID,
    APDCharacteristics,
    APDGeometry,
    CCDGeometry,
    Characteristics,
    ChargeToVoltSettings,
    CMOSGeometry,
    Detector,
    Environment,
    MKIDGeometry,
)
from pyxel.detectors.apd import AvalancheSettings, ConverterFunction, ConverterValues

# This is equivalent to 'import asdf'
asdf = pytest.importorskip(
    "asdf",
    reason="Package 'asdf' is not installed. Use 'pip install asdf'",
)


@pytest.fixture(
    params=(
        "ccd_basic",
        "ccd_100x120",
        "cmos_basic",
        "cmos_100x120",
        "mkid_basic",
        "mkid_100x120",
        "apd_basic",
        "apd_100x120",
    )
)
def detector(request) -> CCD | CMOS | MKID | APD:
    """Create a valid detector."""
    if request.param == "ccd_basic":
        return CCD(
            geometry=CCDGeometry(row=4, col=5),
            environment=Environment(),
            characteristics=Characteristics(),
        )
    elif request.param == "ccd_100x120":
        return CCD(
            geometry=CCDGeometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
            ),
            environment=Environment(temperature=100.1),
            characteristics=Characteristics(
                quantum_efficiency=0.1,
                charge_to_volt=ChargeToVoltSettings(value=0.2),
                pre_amplification=3.3,
                full_well_capacity=10,
            ),
        )
    elif request.param == "cmos_basic":
        return CMOS(
            geometry=CMOSGeometry(row=4, col=5),
            environment=Environment(),
            characteristics=Characteristics(),
        )
    elif request.param == "cmos_100x120":
        return CMOS(
            geometry=CMOSGeometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
            ),
            environment=Environment(temperature=100.1),
            characteristics=Characteristics(
                quantum_efficiency=0.1,
                charge_to_volt=ChargeToVoltSettings(value=0.2),
                pre_amplification=3.3,
                full_well_capacity=10,
            ),
        )
    elif request.param == "mkid_basic":
        return MKID(
            geometry=MKIDGeometry(row=4, col=5),
            environment=Environment(),
            characteristics=Characteristics(),
        )
    elif request.param == "mkid_100x120":
        return MKID(
            geometry=MKIDGeometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
            ),
            environment=Environment(temperature=100.1),
            characteristics=Characteristics(
                quantum_efficiency=0.1,
                charge_to_volt=ChargeToVoltSettings(value=0.2),
                pre_amplification=3.3,
                full_well_capacity=10,
            ),
        )
    elif request.param == "apd_basic":
        return APD(
            geometry=APDGeometry(row=4, col=5),
            environment=Environment(),
            characteristics=APDCharacteristics(
                roic_gain=1.0,
                bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
                avalanche_settings=AvalancheSettings(
                    avalanche_gain=2.0,
                    pixel_reset_voltage=3.0,
                    gain_to_bias=ConverterFunction("lambda gain: 0.15 * gain + 2.5"),
                    bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
                ),
            ),
        )
    elif request.param == "apd_100x120":
        return APD(
            geometry=APDGeometry(
                row=100,
                col=120,
                total_thickness=123.1,
                pixel_horz_size=12.4,
                pixel_vert_size=34.5,
            ),
            environment=Environment(temperature=100.1),
            characteristics=APDCharacteristics(
                quantum_efficiency=0.1,
                full_well_capacity=10,
                adc_bit_resolution=16,
                adc_voltage_range=(0.0, 5.0),
                roic_gain=4.1,
                bias_to_node=ConverterValues([(2.65, 73.7), (4.0, 60.0)]),
                avalanche_settings=AvalancheSettings(
                    avalanche_gain=1.0,
                    pixel_reset_voltage=12.0,
                    gain_to_bias=ConverterFunction("lambda gain: 0.15 * gain + 2.5"),
                    bias_to_gain=ConverterValues([(2.65, 1.0), (4.0, 10.0)]),
                ),
            ),
        )
    else:
        raise NotImplementedError


def test_equal(detector: CCD | CMOS | MKID | APD):
    new_detector = copy.deepcopy(detector)

    assert new_detector.geometry is not detector.geometry
    assert new_detector.geometry == detector.geometry

    assert new_detector.environment is not detector.environment
    assert new_detector.environment == detector.environment

    assert new_detector.characteristics is not detector.characteristics
    assert new_detector.characteristics == detector.characteristics

    assert new_detector is not detector
    assert new_detector == detector


def test_to_from_hdf5(detector: CCD | CMOS | MKID | APD, tmp_path: Path):
    """Test methods `Detector.to_hdf5' and `Detector.from_hdf5`."""
    filename = tmp_path / f"{detector.__class__.__name__}.h5"

    # Save to 'hdf5'
    detector.to_hdf5(filename)

    # Load from 'hdf5
    new_detector = Detector.from_hdf5(filename)

    # Comparisons
    assert new_detector.geometry == detector.geometry
    assert new_detector.environment == detector.environment
    assert new_detector.characteristics == detector.characteristics

    assert new_detector == detector


def test_to_from_asdf(detector: CCD | CMOS | MKID | APD, tmp_path: Path):
    """Test methods `Detector.to_asdf' and `Detector.from_asdf`."""
    filename = tmp_path / f"{detector.__class__.__name__}.asdf"

    # Save to 'asdf'
    detector.to_asdf(filename)

    # Load from 'asdf'
    new_detector = Detector.from_asdf(filename)

    # Comparisons
    assert new_detector.geometry == detector.geometry
    assert new_detector.environment == detector.environment
    assert new_detector.characteristics == detector.characteristics

    assert new_detector == detector


@pytest.mark.parametrize(
    "filename",
    [
        "detector.h5",
        "detector.hdf",
        "detector.asdf",
        Path("detector.hdf5"),
    ],
)
def test_to_from_filename(
    detector: CCD | CMOS | MKID | APD, tmp_path: Path, filename: str
):
    """Test methods `Detector.load` and `Detector.save`."""
    full_filename: Path = tmp_path / filename

    # Save
    detector.save(full_filename)

    # Load
    new_detector = Detector.load(full_filename)

    # Comparisons
    assert new_detector.geometry == detector.geometry
    assert new_detector.environment == detector.environment
    assert new_detector.characteristics == detector.characteristics

    assert new_detector == detector
