#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from dataclasses import dataclass

import numpy as np
import pytest

from pyxel.detectors import (
    CCD,
    CCDGeometry,
    Channels,
    Characteristics,
    Environment,
    Matrix,
    ReadoutPosition,
)
from pyxel.models.charge_measurement import ac_crosstalk, dc_crosstalk
from pyxel.models.charge_measurement.amplifier_crosstalk import (
    convert_matrix_crosstalk,
    convert_readout_crosstalk,
)


@pytest.fixture
def ccd_8x8() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=8,
            col=8,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )
    detector.signal.array = np.zeros(detector.geometry.shape, dtype=float)
    return detector


@pytest.mark.parametrize("mode", ["ac", "dc"])
@pytest.mark.parametrize(
    "coupling_matrix, channel_matrix, readout_directions",
    [
        pytest.param(
            [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]],
            [1, 2, 3, 4],
            [1, 2, 1, 2],
            id="4 channels",
        ),
        pytest.param([[1, 0.5], [0.5, 1]], [1, 2], [1, 2], id="2 channels"),
        pytest.param([[1]], [1], [1], id="1 channel"),
    ],
)
def test_ac_dc_crosstalk(
    ccd_8x8: CCD,
    mode: str,
    coupling_matrix: list,
    channel_matrix: list,
    readout_directions: list,
):
    """Test model 'dc_crosstalk' with valid parameters."""
    if mode == "ac":
        ac_crosstalk(
            detector=ccd_8x8,
            coupling_matrix=coupling_matrix,
            channel_matrix=channel_matrix,
            readout_directions=readout_directions,
        )
    elif mode == "dc":
        dc_crosstalk(
            detector=ccd_8x8,
            coupling_matrix=coupling_matrix,
            channel_matrix=channel_matrix,
            readout_directions=readout_directions,
        )
    else:
        raise NotImplementedError


@dataclass
class OldFormat:
    channel_matrix: list
    readout_directions: list


@pytest.mark.parametrize(
    "mode",
    ["ac", "dc"],
)
@pytest.mark.parametrize(
    "coupling_matrix, old_format, channels",
    [
        pytest.param(
            [
                [1, 0.5, 0, 0],
                [0.5, 1, 0, 0],
                [0, 0, 1, 0.5],
                [0, 0, 0.5, 1],
            ],
            OldFormat(channel_matrix=[1, 2, 3, 4], readout_directions=[1, 2, 1, 2]),
            Channels(
                matrix=Matrix([["OP9", "OP13"], ["OP1", "OP5"]]),
                readout_position=ReadoutPosition(
                    {
                        "OP9": "top-left",
                        "OP13": "top-right",
                        "OP1": "top-left",
                        "OP5": "top-right",
                    }
                ),
            ),
            id="4 channels",
        )
    ],
)
def test_compare_with_and_without_channels(
    mode: str,
    coupling_matrix: list,
    old_format: OldFormat,
    channels: Matrix,
):
    """Compare with and without channels."""
    detector = CCD(
        geometry=CCDGeometry(
            row=100,
            col=100,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )

    # Inject a signal
    rng = np.random.default_rng(seed=12345)
    input_signal = rng.random(size=detector.geometry.shape) * 20.0

    detector.signal.array = input_signal.copy()

    # Process using the 'old' representation of channels
    if mode == "ac":
        ac_crosstalk(
            detector=detector,
            coupling_matrix=coupling_matrix,
            channel_matrix=old_format.channel_matrix,
            readout_directions=old_format.readout_directions,
        )
    elif mode == "dc":
        dc_crosstalk(
            detector=detector,
            coupling_matrix=coupling_matrix,
            channel_matrix=old_format.channel_matrix,
            readout_directions=old_format.readout_directions,
        )
    else:
        raise NotImplementedError

    signal_2d = np.array(detector.signal.array)

    # Add 'channels'
    detector.geometry.channels = channels

    # Re-inject the input signal
    detector.signal.array = input_signal.copy()

    # Re-compute with the channels
    if mode == "ac":
        ac_crosstalk(detector=detector, coupling_matrix=coupling_matrix)
    elif mode == "dc":
        dc_crosstalk(detector=detector, coupling_matrix=coupling_matrix)
    else:
        raise NotImplementedError

    # Compare results
    new_signal_2d = np.array(detector.signal.array)

    np.testing.assert_allclose(signal_2d, new_signal_2d)


# TODO: Add more tests with a 2D 'channel_matrix'
# TODO: Add more tests with file(s)
@pytest.mark.parametrize("mode", ["ac", "dc"])
@pytest.mark.parametrize(
    "coupling_matrix, channel_matrix, readout_directions, exp_exc, exp_msg",
    [
        pytest.param(
            [1, 0.5, 0, 0, 0.5, 1, 0, 0, 0, 0, 1, 0.5, 0, 0, 0.5, 1],
            [1, 2, 3, 4],
            [1, 2, 1, 2],
            ValueError,
            "Expecting 2D 'coupling_matrix'",
            id="1D coupling_matrix",
        ),
        pytest.param(
            [[1, 0.5], [0.5, 1], [0, 0], [0, 0]],
            [1, 2, 3, 4],
            [1, 2, 1, 2],
            ValueError,
            "Expecting a matrix of 4x4 elements for 'coupling_matrix'",
            id="2x4 coupling_matrix",
        ),
        pytest.param(
            [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]],
            [1, 2, 3],
            [1, 2, 1],
            ValueError,
            "Can't split detector array horizontally for a given number of amplifiers",
            id="3 channels",
        ),
        pytest.param(
            [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]],
            [1, 2, 3, 4],
            [1, 2, 1],
            ValueError,
            "Channel matrix and readout directions arrays not the same size",
            id="Different size",
        ),
    ],
)
def test_ac_dc_crosstalk_invalid_params(
    ccd_8x8: CCD,
    mode: str,
    coupling_matrix: list,
    channel_matrix: list,
    readout_directions: list,
    exp_exc,
    exp_msg: str,
):
    """Test model 'dc_crosstalk' with invalid parameters."""
    if mode == "ac":
        with pytest.raises(exp_exc, match=exp_msg):
            ac_crosstalk(
                detector=ccd_8x8,
                coupling_matrix=coupling_matrix,
                channel_matrix=channel_matrix,
                readout_directions=readout_directions,
            )
    elif mode == "dc":
        with pytest.raises(exp_exc, match=exp_msg):
            dc_crosstalk(
                detector=ccd_8x8,
                coupling_matrix=coupling_matrix,
                channel_matrix=channel_matrix,
                readout_directions=readout_directions,
            )
    else:
        raise NotImplementedError


@pytest.mark.parametrize(
    "matrix, exp_array",
    [
        pytest.param(
            Matrix([["OP9", "OP13"], ["OP1", "OP5"]]),
            np.array([1, 2, 3, 4], dtype=int),
            id="2x2",
        ),
        pytest.param(
            Matrix([["OP9", "OP13", "OP1", "OP5"]]),
            np.array([1, 2, 3, 4], dtype=int),
            id="1x4",
        ),
        pytest.param(
            Matrix([["OP9"], ["OP13"], ["OP1"], ["OP5"]]),
            np.array([1, 2, 3, 4], dtype=int),
            id="4x1",
        ),
    ],
)
def test_convert_matrix_crosstalk(matrix: Matrix, exp_array: np.ndarray):
    """Test function 'convert_matrix_crosstalk'."""
    data = convert_matrix_crosstalk(matrix)

    assert isinstance(data, np.ndarray)
    np.testing.assert_allclose(data, exp_array)


@pytest.mark.parametrize(
    "channels, exp_array",
    [
        pytest.param(
            Channels(
                matrix=Matrix([["OP9", "OP13"], ["OP1", "OP5"]]),
                readout_position=ReadoutPosition(
                    {
                        "OP9": "top-left",
                        "OP13": "top-right",
                        "OP1": "bottom-left",
                        "OP5": "bottom-right",
                    }
                ),
            ),
            np.array([1, 2, 3, 4]),
            id="2x2",
        ),
        pytest.param(
            Channels(
                matrix=Matrix([["OP5", "OP1"], ["OP13", "OP9"]]),
                readout_position=ReadoutPosition(
                    {
                        "OP9": "top-left",
                        "OP13": "top-right",
                        "OP1": "bottom-left",
                        "OP5": "bottom-right",
                    }
                ),
            ),
            np.array([4, 3, 2, 1]),
            id="2x2 reversed",
        ),
    ],
)
def test_convert_readout_crosstalk(channels: Channels, exp_array: np.ndarray):
    """Test function 'convert_readout_crosstalk'."""
    data = convert_readout_crosstalk(channels)

    assert isinstance(data, np.ndarray)
    np.testing.assert_allclose(data, exp_array)
