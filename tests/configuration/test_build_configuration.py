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

from pyxel import Configuration, build_configuration, run_mode_dataset


@pytest.fixture(
    params=[
        "CCD",
        # "CMOS", "APD"
    ]
)
def valid_configuration(request) -> Configuration:
    return build_configuration(detector_type=request.param, num_cols=10, num_rows=10)


def test_build_configuration(valid_configuration: Configuration):
    config = valid_configuration
    assert isinstance(config, Configuration)

    ds = run_mode_dataset(config)
    assert isinstance(ds, xr.Dataset)

    assert not np.allclose(ds["photon"], np.zeros((1, 10, 10), dtype=float))
    assert not np.allclose(ds["charge"], np.zeros((1, 10, 10), dtype=float))
    assert not np.allclose(ds["pixel"], np.zeros((1, 10, 10), dtype=float))
    assert not np.allclose(ds["signal"], np.zeros((1, 10, 10), dtype=float))
    assert not np.allclose(ds["image"], np.zeros((1, 10, 10), dtype=float))


def test_to_yaml_to_str(valid_configuration: Configuration):
    config = valid_configuration
    assert isinstance(config, Configuration)

    result = config.to_yaml()
    assert isinstance(result, str)


def test_to_yaml_to_filename(valid_configuration: Configuration, tmp_path: Path):
    config = valid_configuration
    assert isinstance(config, Configuration)

    filename = tmp_path / "config.yaml"
    assert not filename.exists()

    result = config.to_yaml(filename)
    assert result is None

    assert filename.exists()
