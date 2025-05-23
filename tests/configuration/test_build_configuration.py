#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import numpy as np
import pytest
import xarray as xr

from pyxel import Configuration, build_configuration, run_mode_dataset


@pytest.mark.parametrize(
    "detector_type",
    [
        "CCD",
        # "CMOS", "APD"
    ],
)
def test_build_configuration(detector_type: str):
    config = build_configuration(detector_type=detector_type, num_cols=10, num_rows=10)

    assert isinstance(config, Configuration)

    ds = run_mode_dataset(config)
    assert isinstance(ds, xr.Dataset)

    assert not np.allclose(ds["photon"], np.zeros((1, 10, 10), dtype=float))
    assert not np.allclose(ds["charge"], np.zeros((1, 10, 10), dtype=float))
    assert not np.allclose(ds["pixel"], np.zeros((1, 10, 10), dtype=float))
    assert not np.allclose(ds["signal"], np.zeros((1, 10, 10), dtype=float))
    assert not np.allclose(ds["image"], np.zeros((1, 10, 10), dtype=float))
