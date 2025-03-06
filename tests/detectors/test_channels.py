#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import numpy as np
import pytest

from pyxel.detectors.channels import Channels


@pytest.mark.parametrize(
    "matrix, readout_position",
    [
        pytest.param(
            [["OP9", "OP13"], ["OP1", "OP5"]],
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",
                "OP5": "bottom-left",
            },
            id="Grid structure (str)",
        ),
        pytest.param(
            [[9, 13], [1, 5]],
            {9: "top-left", 13: "top-left", 1: "bottom-left", 5: "bottom-left"},
            id="Grid structure (int)",
        ),
        pytest.param(
            [["OP9", 13], [1, "OP5"]],
            {
                "OP9": "top-left",
                13: "top-left",
                1: "bottom-left",
                "OP5": "bottom-left",
            },
            id="Grid structure (str and int)",
        ),
        pytest.param(
            ["OP9", "OP13"],
            {"OP9": "top-left", "OP13": "top-left"},
            id="row structure",
        ),
        pytest.param(
            [["OP9"], ["OP1"]],
            {"OP9": "top-left", "OP1": "bottom-left"},
            id="column structure",
        ),
    ],
)
def test_channels_valid_initialization(matrix, readout_position):
    """Test Channels initialization with correct matrix and readout_position."""
    # Should not raise an error
    channels = Channels(matrix=matrix, readout_position=readout_position)

    # Check if attributes are correctly assigned
    assert isinstance(channels.matrix, np.ndarray)
    assert channels.matrix.tolist() == matrix
    assert channels.readout_position.positions == readout_position


@pytest.mark.parametrize(
    "matrix, readout_position, exp_msg",
    [
        pytest.param(
            [["OP9", "OP13"], ["OP1", "OP5"]],
            {"OP9": "top-left", "OP13": "top-left"},
            r"Readout direction of at least one channel is missing.",
            id="Grid - Missing OP1 and OP5",
        ),
        pytest.param(
            [["OP9", "OP13"], ["OP1", "OP5"]],
            {"OP9": "top-left", "OP13": "top-left", "OP1": "bottom-left"},
            r"Readout direction of at least one channel is missing.",
            id="Grid - Missing OP5",
        ),
        pytest.param(
            [["OP9", "OP13"]],
            {"OP9": "top-left"},
            r"Readout direction of at least one channel is missing.",
            id="Row - Missing OP13",
        ),
        pytest.param(
            [["OP9"], ["OP1"]],
            {"OP1": "top-left"},
            r"Readout direction of at least one channel is missing.",
            id="Column - Missing OP9",
        ),
    ],
)
def test_channels_missing_readout_position(matrix, readout_position, exp_msg):
    """Test Channels raises an error when readout_position is incomplete."""
    with pytest.raises(ValueError, match=exp_msg):
        Channels(matrix=matrix, readout_position=readout_position)


@pytest.mark.parametrize(
    "matrix, readout_position, exp_msg",
    [
        pytest.param(
            [["OP9", "OP13"], ["OP1", "OP5"]],
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",
                "OP5": "bottom-left",
                "OP_EXTRA": "bottom-right",  # Extra key that isn't in matrix
            },
            r"Readout position contains extra channels not listed in matrix.",
            id="Grid - with extra OP_EXTRA",
        ),
        pytest.param(
            ["OP9", "OP13"],
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",  # Extra
                "OP5": "bottom-left",  # Extra
            },
            r"Readout position contains extra channels not listed in matrix.",
            id="Row - with extra OP1 and OP5",
        ),
        pytest.param(
            [["OP9"], ["OP1"]],
            {
                "OP9": "top-left",
                "OP13": "top-left",  # Extra
                "OP1": "bottom-left",
                "OP5": "bottom-left",  # Extra
            },
            r"Readout position contains extra channels not listed in matrix.",
            id="Column - with extra OP13 and OP5",
        ),
    ],
)
def test_channels_extra_readout_position(matrix, readout_position, exp_msg):
    """Test Channels raises an error when readout_position has extra keys."""
    with pytest.raises(ValueError, match=exp_msg):
        Channels(matrix=matrix, readout_position=readout_position)


def test_channels_same_count_different_names():
    """Test Channels raises an error when readout_position has the same count but different names."""
    matrix = [["OP9", "OP13"], ["OP1", "OP5"]]
    readout_position = {
        "X1": "top-left",
        "X2": "top-left",
        "X3": "bottom-left",
        "X4": "bottom-left",
    }

    with pytest.raises(
        ValueError,
        match="Channel names in the matrix and in the readout directions are not matching.",
    ):
        Channels(matrix=matrix, readout_position=readout_position)


@pytest.mark.parametrize(
    "matrix, readout_position",
    [
        pytest.param(
            [["OP9"], ["OP13", "OP1", "OP5"]],
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",
                "OP5": "bottom-left",
            },
            id="Grid structure",
        ),
        pytest.param(
            [["OP9"], ["OP13", "OP1"], ["OP5"]],
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",
                "OP5": "bottom-left",
            },
            id="Grid structure2",
        ),
        pytest.param(
            [[9.0, 13.0]],
            {9.0: "top-left", 13.0: "top-left"},
            id="Row (float)",
        ),
        pytest.param(
            [[9.9, 13.13]],
            {9.9: "top-left", 13.13: "top-left"},
            id="Row2 (float)",
        ),
    ],
)
def test_channels_bad_matrix(matrix, readout_position):
    """Test with a malformed 'matrix' parameter."""
    with pytest.raises(ValueError, match=r"Parameter 'matrix' is malformed"):
        _ = Channels(matrix=matrix, readout_position=readout_position)


@pytest.mark.parametrize(
    "matrix, readout_position, exp_error",
    [
        pytest.param(
            ["OP9", "OP13"],
            {"OP9": "topleft", "OP13": "top-left"},
            r"Did you mean: \'top-left'\?",
            id="row",
        ),
        pytest.param(
            [["OP9"], ["OP13"]],
            {"OP9": "top-left", "OP13": "top-Right"},
            r"Did you mean: \'top-right'\?",
            id="column",
        ),
    ],
)
def test_channels_bad_readout_position(matrix, readout_position, exp_error):
    """Test with a malformed 'readout_position' parameter."""
    with pytest.raises(ValueError, match=exp_error):
        _ = Channels(matrix=matrix, readout_position=readout_position)


def test_eq():
    """Test method '__eq__'."""
    channels1_grid = Channels(
        matrix=[["OP9", "OP13"], ["OP1", "OP5"]],
        readout_position={
            "OP9": "top-left",
            "OP13": "top-left",
            "OP1": "bottom-left",
            "OP5": "bottom-left",
        },
    )

    channels2_grid = Channels(
        matrix=[["OP13", "OP9"], ["OP1", "OP5"]],
        readout_position={
            "OP9": "top-left",
            "OP13": "top-left",
            "OP1": "bottom-left",
            "OP5": "bottom-left",
        },
    )

    channels1_row = Channels(
        matrix=["OP9", "OP13", "OP1", "OP5"],
        readout_position={
            "OP9": "top-left",
            "OP13": "top-left",
            "OP1": "bottom-left",
            "OP5": "bottom-left",
        },
    )

    channels1_column = Channels(
        matrix=[["OP13"], ["OP9"], ["OP1"], ["OP5"]],
        readout_position={
            "OP9": "top-left",
            "OP13": "top-left",
            "OP1": "bottom-left",
            "OP5": "bottom-left",
        },
    )

    assert channels1_grid == deepcopy(channels1_grid)
    assert channels2_grid == deepcopy(channels2_grid)
    assert channels1_row == deepcopy(channels1_row)
    assert channels1_column == deepcopy(channels1_column)

    assert channels1_grid != channels2_grid
    assert channels1_grid != channels1_row
    assert channels1_row != channels1_column


@pytest.mark.parametrize(
    "channels, exp_repr",
    [
        pytest.param(
            Channels(
                matrix=[["OP9", "OP13"], ["OP1", "OP5"]],
                readout_position={
                    "OP9": "top-left",
                    "OP13": "top-left",
                    "OP1": "bottom-left",
                    "OP5": "bottom-left",
                },
            ),
            "Channels<4 channels>",
            id="Grid",
        ),
        pytest.param(
            Channels(
                matrix=["OP9", "OP13"],
                readout_position={
                    "OP9": "top-left",
                    "OP13": "top-left",
                },
            ),
            "Channels<2 channels>",
            id="Row",
        ),
        pytest.param(
            Channels(
                matrix=[["OP9"], ["OP1"]],
                readout_position={
                    "OP9": "top-left",
                    "OP1": "bottom-left",
                },
            ),
            "Channels<2 channels>",
            id="Column",
        ),
    ],
)
def test_repr(channels, exp_repr):
    """Test method '.__repr__'."""
    assert repr(channels) == exp_repr


@pytest.mark.parametrize(
    "matrix, readout_position",
    [
        pytest.param(
            [["OP9", "OP13"], ["OP1", "OP5"]],
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",
                "OP5": "bottom-left",
            },
            id="Grid",
        ),
        pytest.param(
            ["OP9", "OP13"],
            {"OP9": "top-left", "OP13": "top-left"},
            id="Row",
        ),
        pytest.param(
            [["OP9"], ["OP1"]],
            {"OP9": "top-left", "OP1": "bottom-left"},
            id="Column",
        ),
    ],
)
def test_to_dict(matrix, readout_position):
    """Test method '.to_dict'."""
    exp_dct = deepcopy({"matrix": matrix, "readout_position": readout_position})

    # Build a 'Channels' object
    channels = Channels(matrix=matrix, readout_position=readout_position)

    # Check '.to_dict'
    dct = channels.to_dict()
    assert dct == exp_dct
    assert dct["matrix"] is not matrix
    assert dct["readout_position"] is not readout_position


@pytest.mark.parametrize(
    "dct, exp_channels",
    [
        pytest.param(
            {
                "matrix": [["OP9", "OP13"], ["OP1", "OP5"]],
                "readout_position": {
                    "OP9": "top-left",
                    "OP13": "top-left",
                    "OP1": "bottom-left",
                    "OP5": "bottom-left",
                },
            },
            Channels(
                matrix=[["OP9", "OP13"], ["OP1", "OP5"]],
                readout_position={
                    "OP9": "top-left",
                    "OP13": "top-left",
                    "OP1": "bottom-left",
                    "OP5": "bottom-left",
                },
            ),
            id="Grid",
        ),
    ],
)
def test_from_dict(dct, exp_channels):
    """Test method '.from_dict'."""
    channels = Channels.from_dict(dct)

    assert channels == exp_channels
