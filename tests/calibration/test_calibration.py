#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


import logging
from pathlib import Path

import pytest
import xarray as xr

from pyxel.calibration import Algorithm, Calibration, sum_of_abs_residuals
from pyxel.detectors import (
    CCD,
    CCDGeometry,
    Characteristics,
    ChargeToVoltSettings,
    Environment,
)
from pyxel.observation import ParameterValues
from pyxel.pipelines import DetectionPipeline, ModelFunction, Processor

# This is equivalent to 'import pygmo as pg'
pg = pytest.importorskip(
    "pygmo",
    reason="Package 'pygmo' is not installed. Use 'pip install pygmo'",
)


@pytest.fixture
def ccd_detector() -> CCD:
    return CCD(
        geometry=CCDGeometry(row=835, col=1),
        environment=Environment(temperature=238.0),
        characteristics=Characteristics(
            full_well_capacity=90_0000,
            charge_to_volt=ChargeToVoltSettings(value=1e-6),
            adc_bit_resolution=16,
            adc_voltage_range=(0.0, 10.0),
        ),
    )


@pytest.fixture
def pipeline() -> DetectionPipeline:
    return DetectionPipeline(
        charge_generation=[
            ModelFunction(
                func="pyxel.models.charge_generation.load_charge",
                name="load_charge",
                arguments={"filename": "_"},
            )
        ],
        charge_collection=[
            ModelFunction(
                func="pyxel.models.charge_collection.simple_collection",
                name="simple_collection",
            )
        ],
        charge_transfer=[
            ModelFunction(
                func="pyxel.models.charge_transfer.cdm",
                name="cdm",
                arguments={
                    "direction": "parallel",
                    "trap_release_times": [5.0e-3, 5.0e-3, 5.0e-3, 5.0e-3],
                    "trap_densities": [1.0, 1.0, 1.0, 1.0],
                    "sigma": [1.0e-15, 1.0e-15, 1.0e-15, 1.0e-15],
                    "beta": 0.3,  # calibrating this parameter
                    "max_electron_volume": 1.62e-10,  # cm^2
                    "transfer_period": 9.4722e-04,  # s
                    "charge_injection": True,
                },
            )
        ],
        charge_measurement=[
            ModelFunction(
                func="pyxel.models.charge_measurement.simple_measurement",
                name="simple_measurement",
            )
        ],
        readout_electronics=[
            ModelFunction(
                func="pyxel.models.readout_electronics.simple_adc",
                name="simple_adc",
            )
        ],
    )


@pytest.fixture
def processor(ccd_detector: CCD, pipeline: DetectionPipeline) -> Processor:
    return Processor(detector=ccd_detector, pipeline=pipeline)


@pytest.fixture(params=["unconnected", "ring", "fully_connected"])
def calibration(request: pytest.FixtureRequest) -> Calibration:
    topology: str = request.param

    folder = Path("tests/observation")
    assert folder.exists()

    return Calibration(
        target_data_path=[
            folder / "data/target/target_flex_ds7_ch0_1ke.txt",
            folder / "data/target/target_flex_ds7_ch0_3ke.txt",
            folder / "data/target/target_flex_ds7_ch0_7ke.txt",
            folder / "data/target/target_flex_ds7_ch0_10ke.txt",
            folder / "data/target/target_flex_ds7_ch0_20ke.txt",
            folder / "data/target/target_flex_ds7_ch0_100ke.txt",
        ],
        fitness_function=sum_of_abs_residuals,
        algorithm=Algorithm(type="sade", generations=10, population_size=20),
        parameters=[
            ParameterValues(
                key="pipeline.charge_transfer.cdm.arguments.beta",
                values="_",
                logarithmic=False,
                boundaries=(0.1, 0.9),
            ),
            ParameterValues(
                key="pipeline.charge_transfer.cdm.arguments.trap_release_times",
                values=["_", "_", "_", "_"],
                logarithmic=True,
                boundaries=(1e-5, 1e-1),
            ),
            ParameterValues(
                key="pipeline.charge_transfer.cdm.arguments.trap_densities",
                values=["_", "_", "_", "_"],
                logarithmic=True,
                boundaries=(1e-2, 1e2),
            ),
        ],
        result_input_arguments=[
            ParameterValues(
                key="pipeline.charge_generation.load_charge.arguments.filename",
                values=[
                    "tests/observation/data/input/input_flex_ds7_ch0_1ke.txt",
                    "tests/observation/data/input/input_flex_ds7_ch0_3ke.txt",
                    "tests/observation/data/input/input_flex_ds7_ch0_7ke.txt",
                    "tests/observation/data/input/input_flex_ds7_ch0_10ke.txt",
                    "tests/observation/data/input/input_flex_ds7_ch0_20ke.txt",
                    "tests/observation/data/input/input_flex_ds7_ch0_100ke.txt",
                ],
            ),
        ],
        result_fit_range=(500, 835, 0, 1),
        target_fit_range=(500, 835, 0, 1),
        topology=topology,
    )


@pytest.mark.parametrize(
    "value",
    ["scene", "photon", "charge", "pixel", "signal", "image", "data", "all"],
)
def test_calibration_result_type(calibration: Calibration, value: str):
    """Test class 'Calibration'."""
    calibration.result_type = value
    assert calibration.result_type == value


def test_run_calibration(
    caplog,
    tmp_path: Path,
    calibration: Calibration,
    processor: Processor,
):
    """Test method 'Calibration.run_calibration'."""
    caplog.set_level(logging.INFO)
    dt = calibration.run_calibration(
        processor=processor,
        output_dir=tmp_path,
        with_inherited_coords=False,
    )
    assert isinstance(dt, xr.DataTree)
