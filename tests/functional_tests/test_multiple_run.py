#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import numpy as np
import xarray as xr

import pyxel


def test_multiple_run_only_single_readout():
    """Test multiple consecutive run with a single readout."""

    #
    # Before first run
    #
    cfg = pyxel.load("tests/functional_tests/data/config_2x2.yaml")

    # Check 'detector.charge'
    charge_2d = cfg.detector.charge.array
    assert isinstance(charge_2d, np.ndarray)

    exp_charge_2d = np.array([[0, 0], [0, 0]], dtype=float)
    np.testing.assert_equal(charge_2d, exp_charge_2d)

    #
    # First run
    #
    result = pyxel.run_mode(
        mode=cfg.exposure,
        detector=cfg.detector,
        pipeline=cfg.pipeline,
    )

    # Check 'detector.charge'
    charge_2d = cfg.detector.charge.array
    assert isinstance(charge_2d, np.ndarray)

    exp_charge_2d = np.array([[10, 10], [10, 10]], dtype=float)
    np.testing.assert_equal(charge_2d, exp_charge_2d)

    # Check 'result.charge'
    charge_array_3d = result["/bucket/charge"]
    assert isinstance(charge_array_3d, xr.DataArray)
    exp_charge_array_3d = xr.DataArray(
        np.array([[[10.0, 10.0], [10.0, 10.0]]], dtype=float),
        dims=["time", "y", "x"],
        coords={"time": [10.0], "y": [0, 1], "x": [0, 1]},
        attrs={"units": "e⁻", "long_name": "Charge"},
    )
    xr.testing.assert_equal(charge_array_3d, exp_charge_array_3d)

    #
    # Before second run
    #
    cfg = pyxel.load("tests/functional_tests/data/config_2x2.yaml")

    # Check 'detector.charge'
    charge_2d = cfg.detector.charge.array
    assert isinstance(charge_2d, np.ndarray)

    exp_charge_2d = np.array([[0, 0], [0, 0]], dtype=float)
    np.testing.assert_equal(charge_2d, exp_charge_2d)

    #
    # Second run
    #
    result = pyxel.run_mode(
        mode=cfg.exposure,
        detector=cfg.detector,
        pipeline=cfg.pipeline,
    )

    # Check 'detector.charge'
    charge_2d = cfg.detector.charge.array
    assert isinstance(charge_2d, np.ndarray)

    exp_charge_2d = np.array([[10, 10], [10, 10]], dtype=float)
    np.testing.assert_equal(charge_2d, exp_charge_2d)

    # Check 'result.charge'
    charge_array_3d = result["/bucket/charge"]
    assert isinstance(charge_array_3d, xr.DataArray)
    exp_charge_array_3d = xr.DataArray(
        np.array([[[10.0, 10.0], [10.0, 10.0]]], dtype=float),
        dims=["time", "y", "x"],
        coords={"time": [10.0], "y": [0, 1], "x": [0, 1]},
        attrs={"units": "e⁻", "long_name": "Charge"},
    )
    xr.testing.assert_equal(charge_array_3d, exp_charge_array_3d)


def test_multiple_run_multiple_readout_destructive_readout():
    """Test multiple consecutive run with multiple readouts with destructive readout."""

    cfg = pyxel.load(
        "tests/functional_tests/data/config_multiple_readout_destructive_2x2.yaml"
    )

    result = pyxel.run_mode(
        mode=cfg.exposure,
        detector=cfg.detector,
        pipeline=cfg.pipeline,
    )

    # Check 'result.charge'
    charge_array_3d = result["/bucket/charge"]
    assert isinstance(charge_array_3d, xr.DataArray)
    exp_charge_array_3d = xr.DataArray(
        np.array(
            [
                [[10.0, 10.0], [10.0, 10.0]],
                [[10.0, 10.0], [10.0, 10.0]],
                [[10.0, 10.0], [10.0, 10.0]],
            ],
            dtype=float,
        ),
        dims=["time", "y", "x"],
        coords={"time": [10.0, 20.0, 30.0], "y": [0, 1], "x": [0, 1]},
        attrs={"units": "e⁻", "long_name": "Charge"},
    )
    xr.testing.assert_equal(charge_array_3d, exp_charge_array_3d)

    # Check 'result.pixel'
    pixel_3d = result["/bucket/pixel"]
    assert isinstance(pixel_3d, xr.DataArray)
    exp_pixel_3d = xr.DataArray(
        np.array(
            [
                [[10.0, 10.0], [10.0, 10.0]],
                [[10.0, 10.0], [10.0, 10.0]],
                [[10.0, 10.0], [10.0, 10.0]],
            ],
            dtype=float,
        ),
        dims=["time", "y", "x"],
        coords={"time": [10.0, 20.0, 30.0], "y": [0, 1], "x": [0, 1]},
        attrs={"units": "e⁻", "long_name": "Pixel"},
    )
    xr.testing.assert_equal(pixel_3d, exp_pixel_3d)


def test_multiple_run_multiple_readout_non_destructive_readout():
    """Test multiple consecutive run with multiple readouts with destructive readout."""

    cfg = pyxel.load(
        "tests/functional_tests/data/config_multiple_readout_non_destructive_2x2.yaml"
    )

    result = pyxel.run_mode(
        mode=cfg.exposure,
        detector=cfg.detector,
        pipeline=cfg.pipeline,
    )

    # Check 'result.charge'
    charge_array_3d = result["/bucket/charge"]
    assert isinstance(charge_array_3d, xr.DataArray)
    exp_charge_array_3d = xr.DataArray(
        np.array(
            [
                [[10.0, 10.0], [10.0, 10.0]],
                [[10.0, 10.0], [10.0, 10.0]],
                [[10.0, 10.0], [10.0, 10.0]],
            ],
            dtype=float,
        ),
        dims=["time", "y", "x"],
        coords={"time": [10.0, 20.0, 30.0], "y": [0, 1], "x": [0, 1]},
        attrs={"units": "e⁻", "long_name": "Charge"},
    )
    xr.testing.assert_equal(charge_array_3d, exp_charge_array_3d)

    # Check 'result.pixel'
    pixel_3d = result["/bucket/pixel"]
    assert isinstance(pixel_3d, xr.DataArray)
    exp_pixel_3d = xr.DataArray(
        np.array(
            [
                [[10.0, 10.0], [10.0, 10.0]],
                [[20.0, 20.0], [20.0, 20.0]],
                [[30.0, 30.0], [30.0, 30.0]],
            ],
            dtype=float,
        ),
        dims=["time", "y", "x"],
        coords={"time": [10.0, 20.0, 30.0], "y": [0, 1], "x": [0, 1]},
        attrs={"units": "e⁻", "long_name": "Pixel"},
    )
    xr.testing.assert_equal(pixel_3d, exp_pixel_3d)
