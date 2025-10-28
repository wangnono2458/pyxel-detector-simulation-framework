#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import numpy as np
import pytest

from pyxel.detectors import CMOS, Characteristics, CMOSGeometry, Environment
from pyxel.models.charge_measurement import nghxrg
from pyxel.models.charge_measurement.nghxrg.nghxrg import (
    AcnNoise,
    CorrPinkNoise,
    KTCBiasNoise,
    PCAZeroNoise,
    UncorrPinkNoise,
    WhiteReadNoise,
    _get_noise_type,
    compute_nghxrg,
)


@pytest.fixture
def cmos_10x10() -> CMOS:
    """Create a valid CCD detector."""
    detector = CMOS(
        geometry=CMOSGeometry(
            row=10,
            col=10,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )

    # Generate fake pixels
    rs = np.random.RandomState(12345)
    detector.pixel.non_volatile.array = rs.normal(loc=100, scale=5, size=(10, 10))

    return detector


@pytest.fixture
def cmos_10x15() -> CMOS:
    """Create a valid CCD detector."""
    detector = CMOS(
        geometry=CMOSGeometry(
            row=10,
            col=15,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )

    # Generate fake pixels
    rs = np.random.RandomState(12345)
    detector.pixel.non_volatile.array = rs.normal(loc=10000, scale=500, size=(10, 15))

    return detector


def test_nghxrg_10x10_full(cmos_10x10: CMOS):
    """Test model 'nghxrg' without parameters."""
    detector = cmos_10x10

    noise = [
        {"ktc_bias_noise": {"ktc_noise": 1, "bias_offset": 2, "bias_amp": 2}},
        {"white_read_noise": {"rd_noise": 1, "ref_pixel_noise_ratio": 2}},
        {"corr_pink_noise": {"c_pink": 1.0}},
        {"uncorr_pink_noise": {"u_pink": 1.0}},
        {"acn_noise": {"acn": 1.0}},
        {"pca_zero_noise": {"pca0_amp": 1.0}},
    ]

    nghxrg(detector=detector, noise=noise)


@pytest.mark.parametrize("n_output", [-1, 33])
def test_nghxrg_bad_noutput(cmos_10x10: CMOS, n_output):
    """Test model 'nghxrg' with bad inputs."""
    detector = cmos_10x10

    noise = [
        {"ktc_bias_noise": {"ktc_noise": 1, "bias_offset": 2, "bias_amp": 2}},
        {"white_read_noise": {"rd_noise": 1, "ref_pixel_noise_ratio": 2}},
        {"corr_pink_noise": {"c_pink": 1.0}},
        {"uncorr_pink_noise": {"u_pink": 1.0}},
        {"acn_noise": {"acn": 1.0}},
        {"pca_zero_noise": {"pca0_amp": 1.0}},
    ]

    with pytest.raises(ValueError, match="'n_output' must be between 0 and 32"):
        nghxrg(detector=detector, noise=noise, n_output=n_output)


@pytest.mark.parametrize("n_row_overhead", [-1, 101])
def test_nghxrg_bad_n_row_overhead(cmos_10x10: CMOS, n_row_overhead):
    """Test model 'nghxrg' with bad inputs."""
    detector = cmos_10x10

    noise = [
        {"ktc_bias_noise": {"ktc_noise": 1, "bias_offset": 2, "bias_amp": 2}},
        {"white_read_noise": {"rd_noise": 1, "ref_pixel_noise_ratio": 2}},
        {"corr_pink_noise": {"c_pink": 1.0}},
        {"uncorr_pink_noise": {"u_pink": 1.0}},
        {"acn_noise": {"acn": 1.0}},
        {"pca_zero_noise": {"pca0_amp": 1.0}},
    ]

    with pytest.raises(ValueError, match="'n_row_overhead' must be between 0 and 100"):
        nghxrg(detector=detector, noise=noise, n_row_overhead=n_row_overhead)


@pytest.mark.parametrize("n_frame_overhead", [-1, 101])
def test_nghxrg_bad_n_frame_overhead(cmos_10x10: CMOS, n_frame_overhead):
    """Test model 'nghxrg' with bad inputs."""
    detector = cmos_10x10

    noise = [
        {"ktc_bias_noise": {"ktc_noise": 1, "bias_offset": 2, "bias_amp": 2}},
        {"white_read_noise": {"rd_noise": 1, "ref_pixel_noise_ratio": 2}},
        {"corr_pink_noise": {"c_pink": 1.0}},
        {"uncorr_pink_noise": {"u_pink": 1.0}},
        {"acn_noise": {"acn": 1.0}},
        {"pca_zero_noise": {"pca0_amp": 1.0}},
    ]

    with pytest.raises(
        ValueError, match="'n_frame_overhead' must be between 0 and 100"
    ):
        nghxrg(detector=detector, noise=noise, n_frame_overhead=n_frame_overhead)


@pytest.mark.parametrize("reference_pixel_border_width", [-1, 33])
def test_nghxrg_reference_pixel_border_width(
    cmos_10x10: CMOS, reference_pixel_border_width
):
    """Test model 'nghxrg' with bad inputs."""
    detector = cmos_10x10

    noise = [
        {"ktc_bias_noise": {"ktc_noise": 1, "bias_offset": 2, "bias_amp": 2}},
        {"white_read_noise": {"rd_noise": 1, "ref_pixel_noise_ratio": 2}},
        {"corr_pink_noise": {"c_pink": 1.0}},
        {"uncorr_pink_noise": {"u_pink": 1.0}},
        {"acn_noise": {"acn": 1.0}},
        {"pca_zero_noise": {"pca0_amp": 1.0}},
    ]

    with pytest.raises(
        ValueError, match="'reference_pixel_border_width' must be between 0 and 32"
    ):
        nghxrg(
            detector=detector,
            noise=noise,
            reference_pixel_border_width=reference_pixel_border_width,
        )


# TODO: This test fails because 'y' and 'x' are not in the right order
def test_nghxrg_10x15(cmos_10x15: CMOS):
    """Test model 'nghxrg' without parameters."""

    noise = [
        {"ktc_bias_noise": {"ktc_noise": 1, "bias_offset": 2, "bias_amp": 2}},
        {"white_read_noise": {"rd_noise": 1, "ref_pixel_noise_ratio": 2}},
        {"corr_pink_noise": {"c_pink": 1.0}},
        {"uncorr_pink_noise": {"u_pink": 1.0}},
        {"acn_noise": {"acn": 1.0}},
        {"pca_zero_noise": {"pca0_amp": 1.0}},
    ]

    nghxrg(detector=cmos_10x15, noise=noise)


def test_compute_nghxrg_full():
    """Test function 'compute_nghxrg'."""
    noise = (
        KTCBiasNoise(ktc_noise=1, bias_offset=2, bias_amp=2),
        WhiteReadNoise(rd_noise=1, ref_pixel_noise_ratio=2),
        CorrPinkNoise(c_pink=1.0),
        UncorrPinkNoise(u_pink=1.0),
        AcnNoise(acn=1.0),
        PCAZeroNoise(pca0_amp=1.0),
    )

    rs = np.random.RandomState(12345)
    pixel_2d = rs.normal(loc=100, scale=5, size=(10, 10))

    result = compute_nghxrg(
        pixel_2d=pixel_2d,
        noise=noise,
        detector_shape=(10, 10),
        window_pos=(0, 0),
        window_size=(10, 10),
        num_outputs=1,
        time_step=1,
        num_rows_overhead=0,
        num_frames_overhead=0,
        reverse_scan_direction=False,
        reference_pixel_border_width=4,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)


def test_compute_nghxrg_window():
    """Test function 'compute_nghxrg'."""
    noise = (
        KTCBiasNoise(ktc_noise=1, bias_offset=2, bias_amp=2),
        WhiteReadNoise(rd_noise=1, ref_pixel_noise_ratio=2),
        CorrPinkNoise(c_pink=1.0),
        UncorrPinkNoise(u_pink=1.0),
        AcnNoise(acn=1.0),
        PCAZeroNoise(pca0_amp=1.0),
    )

    rs = np.random.RandomState(12345)
    pixel_2d = rs.normal(loc=100, scale=5, size=(20, 20))

    result = compute_nghxrg(
        pixel_2d=pixel_2d,
        noise=noise,
        detector_shape=(10, 10),
        window_pos=(1, 1),
        window_size=(8, 8),
        num_outputs=1,
        time_step=1,
        num_rows_overhead=0,
        num_frames_overhead=0,
        reverse_scan_direction=False,
        reference_pixel_border_width=4,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)


def test_get_noise_type_invalid():
    """Test function '_get_noise_type'."""
    with pytest.raises(KeyError, match="Unknown key"):
        _ = _get_noise_type(item={"foo": "bar"})
