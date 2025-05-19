#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import pyxel
from pyxel import Configuration
from pyxel.detectors import Detector
from pyxel.exposure import Exposure

# Check if 'poppy' is installed
_ = pytest.importorskip(
    "poppy",
    reason="Package 'poppy' is not installed. Use 'pip install poppy'",
)


@pytest.fixture
def valid_config_filename(request: pytest.FixtureRequest) -> Path:
    """Get a valid existing YAML filename."""
    filename: Path = request.path.parent / "data/basic_exposure.yaml"
    return filename.resolve(strict=True)


def test_basic_exposure_hdf5(valid_config_filename: Path, tmp_path: Path):
    """Functional test with a basic Exposure mode."""
    # Read configuration file
    cfg = pyxel.load(valid_config_filename)
    assert isinstance(cfg, Configuration)

    # Save 'detector' before modifications
    detector = cfg.detector
    assert isinstance(detector, Detector)

    detector_filename_before = tmp_path / "before_detector.hdf5"
    assert not detector_filename_before.exists()

    detector.to_hdf5(detector_filename_before)
    assert detector_filename_before.exists()

    # Execute 'cfg'
    data_tree = pyxel.run_mode(
        mode=cfg.running_mode,
        detector=detector,
        pipeline=cfg.pipeline,
    )
    assert isinstance(data_tree, xr.DataTree)

    # Save the 'detector' object into a '.hdf5' file
    detector_filename: Path = tmp_path / "detector.hdf5"
    assert not detector_filename.exists()

    detector.to_hdf5(detector_filename)
    assert detector_filename.exists()

    # Load to a new 'detector' object from '.hdf5' file
    new_detector = Detector.from_hdf5(detector_filename)

    assert detector.data.isomorphic(new_detector.data)
    assert set(detector.data.groups) == set(new_detector.data.groups)
    # assert detector == new_detector


def test_basic_exposure_asdf(valid_config_filename: Path, tmp_path: Path):
    """Functional test with a basic Exposure mode."""
    # Read configuration file
    cfg = pyxel.load(valid_config_filename)
    assert isinstance(cfg, Configuration)

    # Save 'detector' before modifications
    detector = cfg.detector
    assert isinstance(detector, Detector)

    detector_filename_before = tmp_path / "before_detector.asdf"
    assert not detector_filename_before.exists()

    detector.to_asdf(detector_filename_before)
    assert detector_filename_before.exists()

    # Execute 'cfg'
    data_tree = pyxel.run_mode(
        mode=cfg.running_mode,
        detector=cfg.detector,
        pipeline=cfg.pipeline,
    )
    assert isinstance(data_tree, xr.DataTree)

    # Save the 'detector' object into a '.asdf' file
    detector_filename: Path = tmp_path / "detector.asdf"
    assert not detector_filename.exists()

    detector.to_asdf(detector_filename)
    assert detector_filename.exists()

    # Load to a new 'detector' object from '.asdf' file
    new_detector = Detector.from_asdf(detector_filename)

    assert detector.data.isomorphic(new_detector.data)
    assert set(detector.data.groups) == set(new_detector.data.groups)
    # assert detector == new_detector


@pytest.mark.parametrize("readout_times", [[0.1], [0.1, 0.5]])
def test_simple_adc_dtype(readout_times, valid_config_filename: Path):
    """Functional tests to check '.dtype' when using model 'simple_adc'."""
    # Read configuration file
    cfg = pyxel.load(valid_config_filename)
    assert isinstance(cfg, Configuration)

    detector = cfg.detector

    exposure = cfg.running_mode
    assert isinstance(exposure, Exposure)

    exposure.readout.times = readout_times

    result = pyxel.run_mode(
        mode=exposure,
        detector=detector,
        pipeline=cfg.pipeline,
    )

    assert detector.image.dtype == np.uint16
    assert result["image"].dtype == np.uint16


@pytest.mark.parametrize(
    "run_with_config", ["config_and_args", "config_and_kwargs", "without_config"]
)
def test_with_debug(run_with_config: str, valid_config_filename: Path):
    """Test 'pyxel.run_mode' with parameter 'debug' enabled."""
    cfg = pyxel.load(valid_config_filename)

    if run_with_config == "config_and_args":
        data_tree = pyxel.run_mode(cfg)
    elif run_with_config == "config_and_kwargs":
        data_tree = pyxel.run_mode(config=cfg)
    elif run_with_config == "without_config":
        data_tree = pyxel.run_mode(
            mode=cfg.running_mode,
            detector=cfg.detector,
            pipeline=cfg.pipeline,
            debug=True,
        )
    else:
        raise NotImplementedError

    assert isinstance(data_tree, xr.DataTree)
