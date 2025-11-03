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


@pytest.fixture
def cmos_5x5() -> CMOS:
    """Create a valid CCD detector."""
    detector = CMOS(
        geometry=CMOSGeometry(row=5, col=5),
        environment=Environment(),
        characteristics=Characteristics(),
    )

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

    non_volatile = np.array(detector.pixel.non_volatile)
    # fmt:off
    exp_volatile = np.array([
        [-0.65744862, -2.33237013,  4.2124573 ,  1.16960198, -3.84609839, 1.34860843, -1.5698424 , -0.74783042, -0.91416503,  0.79474651],
        [-1.65783092,  3.9194538 , -1.2159954 , -1.91577458, -2.52123778,  0.35789782,  5.39155555,  0.1746172 ,  2.10508665,  3.64896291],
        [ 4.35553938, -1.24009361,  0.54635288, -3.42521464, -2.57524259, -1.85456776, -1.61402779, -1.90387604, -3.08762813, -4.30139617],
        [-0.87759572, -2.07847488, -1.78533792,  2.06895022,  1.0100438 , -2.81076895, -0.68359733,  1.28476231,  5.93676716,  2.15516344],
        [-1.31272939, -2.29833856, -1.35773177, -3.66495788,  1.32930295, -0.66211113,  2.10885838,  5.10844019,  4.80119351, -0.9383803 ],
        [ 2.19907633, -0.3857321 ,  5.71939138,  5.7938083 , -0.17259765,  0.82997602, -2.32774335, -1.52387072,  0.07654423, -2.64097053],
        [ 0.08384882, -2.27563237, -3.1641338 , -4.75678585,  1.04680078,  2.67475008,  3.06660828,  2.25851714,  2.03402572,  2.77779051],
        [ 0.29228224, -5.28256875, -2.31573993,  1.32753906, -1.53473562,  2.9114233 ,  0.60515559,  0.87193531,  1.69864901,  0.24103335],
        [-1.44740438,  0.23182182,  1.78723974, -0.82327522,  3.43254517,  2.72374785, -1.70249677, -3.79172339, -2.98582223, -1.74257395],
        [-0.97747987,  3.69688306,  6.71864194,  1.77425236, -4.77138795,  6.20145635,  2.3336816 , -2.97195977,  5.92882937, -2.71562767]
    ])
    # fmt:off

    # Check before applying the model
    with pytest.raises(ValueError, match=r"not initialized"):
        _ = detector.pixel.volatile.array

    # Run model
    nghxrg(detector=detector, noise=noise, seed=123)

    np.testing.assert_allclose(detector.pixel.non_volatile, non_volatile)
    np.testing.assert_allclose(detector.pixel.volatile, exp_volatile)


@pytest.mark.parametrize("shape", [(7, 8), (8, 7)])
def test_detector_too_small(shape):
    row, col = shape

    detector = CMOS(
        geometry=CMOSGeometry(row=row, col=col),
        environment=Environment(),
        characteristics=Characteristics(),
    )

    # Generate fake pixels
    rs = np.random.RandomState(12345)
    detector.pixel.non_volatile.array = rs.normal(
        loc=100, scale=5, size=detector.geometry.shape
    )

    noise = [
        {"ktc_bias_noise": {"ktc_noise": 1, "bias_offset": 2, "bias_amp": 2}},
        {"white_read_noise": {"rd_noise": 1, "ref_pixel_noise_ratio": 2}},
        {"corr_pink_noise": {"c_pink": 1.0}},
        {"uncorr_pink_noise": {"u_pink": 1.0}},
        {"acn_noise": {"acn": 1.0}},
        {"pca_zero_noise": {"pca0_amp": 1.0}},
    ]

    with pytest.raises(ValueError, match="Geometry too small"):
        nghxrg(detector=detector, noise=noise, seed=123)


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
