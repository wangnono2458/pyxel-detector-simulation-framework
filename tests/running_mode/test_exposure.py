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


@pytest.fixture
def folder_data(request: pytest.FixtureRequest) -> Path:
    """Get the folder 'tests'."""
    folder = Path(request.module.__file__).parent
    return folder.resolve(strict=True)


@pytest.mark.parametrize("with_debug", [False, True])
def test_exposure(folder_data: Path, with_debug: bool, tmp_path: Path):
    """Run 'exposure' mode."""
    filename = folder_data / "data/exposure.yaml"
    assert filename.exists()

    config = pyxel.load(filename)
    config.running_mode.outputs.output_folder = tmp_path

    data_tree = pyxel.run_mode(config, debug=with_debug)

    exp_bucket = xr.Dataset(
        {
            "photon": xr.DataArray(
                [
                    [[124.5110, 124.5110, 124.5110], [124.5110, 124.5110, 124.5110]],
                    [[124.5110, 124.5110, 124.5110], [124.5110, 124.5110, 124.5110]],
                    [[124.5110, 124.5110, 124.5110], [124.5110, 124.5110, 124.5110]],
                ],
                dims=["time", "y", "x"],
            ),
            "charge": xr.DataArray(
                [
                    [[93.0, 101.0, 103.0], [103.0, 99.0, 98.0]],
                    [[91.0, 98.0, 96.0], [98.0, 96.0, 91.0]],
                    [[109.0, 105.0, 102.0], [97.0, 95.0, 94.0]],
                ],
                dims=["time", "y", "x"],
            ),
            "pixel": xr.DataArray(
                [
                    [[93.0, 101.0, 103.0], [103.0, 99.0, 98.0]],
                    [[184.0, 199.0, 199.0], [201.0, 195.0, 189.0]],
                    [[293.0, 304.0, 301.0], [298.0, 290.0, 283.0]],
                ],
                dims=["time", "y", "x"],
            ),
            "signal": xr.DataArray(
                [
                    [[0.0093, 0.0101, 0.0103], [0.0103, 0.0099, 0.0098]],
                    [[0.0184, 0.0199, 0.0199], [0.0201, 0.0195, 0.0189]],
                    [[0.0293, 0.0304, 0.0301], [0.0298, 0.029, 0.0283]],
                ],
                dims=["time", "y", "x"],
            ),
            "image": xr.DataArray(
                np.array(
                    [
                        [[60, 66, 67], [67, 64, 64]],
                        [[120, 130, 130], [131, 127, 123]],
                        [[192, 199, 197], [195, 190, 185]],
                    ],
                    dtype=np.uint16,
                ),
                dims=["time", "y", "x"],
            ),
        },
        coords={"time": [1.0, 2.0, 3.0], "y": [0, 1], "x": [0, 1, 2]},
    )
    xr.testing.assert_allclose(data_tree["/bucket"].to_dataset(), exp_bucket)

    bucket = data_tree["/bucket"].to_dataset()
    xr.testing.assert_allclose(bucket, exp_bucket)
