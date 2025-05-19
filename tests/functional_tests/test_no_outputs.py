#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from pathlib import Path

import pytest

import pyxel


@pytest.fixture
def filename_exposure_no_outputs(request: pytest.FixtureRequest) -> Path:
    """Get a valid existing YAML filename."""
    filename: Path = request.path.parent / "data/simple_exposure_no_outputs.yaml"
    return filename.resolve(strict=True)


@pytest.fixture
def filename_observation_no_outputs(request: pytest.FixtureRequest) -> Path:
    """Get a valid existing YAML filename."""
    filename: Path = request.path.parent / "data/simple_observation_no_outputs.yaml"
    return filename.resolve(strict=True)


@pytest.mark.parametrize(
    "run_with_config", ["config_and_args", "config_and_kwargs", "without_config"]
)
def test_exposure_no_outputs(run_with_config: str, filename_exposure_no_outputs: Path):
    """Test 'pyxel.run_mode' without outputs."""
    cfg = pyxel.load(filename_exposure_no_outputs)

    if run_with_config == "config_and_args":
        _ = pyxel.run_mode(cfg)
    elif run_with_config == "config_and_kwargs":
        _ = pyxel.run_mode(config=cfg)
    elif run_with_config == "without_config":
        _ = pyxel.run_mode(
            mode=cfg.running_mode,
            detector=cfg.detector,
            pipeline=cfg.pipeline,
        )
    else:
        raise NotImplementedError


@pytest.mark.parametrize(
    "run_with_config", ["config_and_args", "config_and_kwargs", "without_config"]
)
def test_observation_no_outputs(
    run_with_config: str, filename_observation_no_outputs: Path
):
    """Test 'pyxel.run_mode' without outputs."""
    cfg = pyxel.load(filename_observation_no_outputs)

    if run_with_config == "config_and_args":
        _ = pyxel.run_mode(cfg)
    elif run_with_config == "config_and_kwargs":
        _ = pyxel.run_mode(config=cfg)
    elif run_with_config == "without_config":
        _ = pyxel.run_mode(
            mode=cfg.running_mode,
            detector=cfg.detector,
            pipeline=cfg.pipeline,
        )
    else:
        raise NotImplementedError
