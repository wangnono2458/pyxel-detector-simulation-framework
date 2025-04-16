#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy

import numpy as np
import pytest

from pyxel.detectors.channels import Channels, Matrix, ReadoutPosition


@pytest.mark.parametrize(
    "matrix, readout_position, exp_matrix",
    [
        pytest.param(
            [["OP9", "OP13"], ["OP1", "OP5"]],
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",
                "OP5": "bottom-left",
            },
            [["OP9", "OP13"], ["OP1", "OP5"]],
            id="Grid structure (str)",
        ),
        pytest.param(
            [[9, 13], [1, 5]],
            {9: "top-left", 13: "top-left", 1: "bottom-left", 5: "bottom-left"},
            [[9, 13], [1, 5]],
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
            [["OP9", 13], [1, "OP5"]],
            id="Grid structure (str and int)",
        ),
        pytest.param(
            [["OP9", "OP13"]],
            {"OP9": "top-left", "OP13": "top-left"},
            [["OP9", "OP13"]],
            id="row structure",
        ),
        pytest.param(
            [["OP9"], ["OP1"]],
            {"OP9": "top-left", "OP1": "bottom-left"},
            [["OP9"], ["OP1"]],
            id="column structure",
        ),
    ],
)
def test_channels_valid_initialization(matrix, readout_position, exp_matrix):
    """Test Channels initialization with correct matrix and readout_position."""
    # Should not raise an error
    channels = Channels(
        matrix=Matrix(matrix), readout_position=ReadoutPosition(readout_position)
    )

    # Check if attributes are correctly assigned
    assert isinstance(channels.matrix, Matrix)
    assert np.array(channels.matrix).tolist() == exp_matrix
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
        Channels(
            matrix=Matrix(matrix), readout_position=ReadoutPosition(readout_position)
        )


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
            [["OP9", "OP13"]],
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
        Channels(
            matrix=Matrix(matrix), readout_position=ReadoutPosition(readout_position)
        )


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
        Channels(
            matrix=Matrix(matrix), readout_position=ReadoutPosition(readout_position)
        )


@pytest.mark.parametrize(
    "data, error_message",
    [
        pytest.param(
            [["OP9"], ["OP13", "OP1", "OP5"]],
            "Parameter 'matrix' is malformed: All rows must be of the same length.",
            id="Mismatched row lengths",
        ),
        pytest.param(
            [["OP9"], ["OP13", "OP1"], ["OP5"]],
            "Parameter 'matrix' is malformed: All rows must be of the same length.",
            id="Each row different lengths",
        ),
        pytest.param(
            ["OP9", "OP13", "OP1", "OP5"],
            "All rows in the matrix must be sequences and not strings.",
            id="Single string inputs treated as rows",
        ),
        pytest.param(
            [], "Matrix data must contain at least one row.", id="Empty matrix"
        ),
        pytest.param(
            [[9, 13], [1]],
            "Parameter 'matrix' is malformed: All rows must be of the same length.",
            id="Numerical row length mismatch",
        ),
        pytest.param(
            [[], []],
            "Parameter 'matrix' is malformed: Cannot have empty rows.",
            id="Empty rows but equal length",
        ),
    ],
)
def test_matrix_initialization(data, error_message):
    if error_message:
        with pytest.raises(ValueError, match=error_message):
            _ = Matrix(data)
    else:
        # Expecting successful initialization here
        matrix = Matrix(data)
        assert isinstance(
            matrix, Matrix
        ), "Matrix instance should be created successfully."


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            [["OP9"], ["OP13"], ["OP1"], ["OP5"]], id="Uniform row lengths of 1"
        ),
        pytest.param([["OP9", "OP13"], ["OP1", "OP5"]], id="Uniform row lengths of 2"),
        pytest.param([[1, 2], [3, 4]], id="Numerical 2x2 matrix"),
        pytest.param([["OP9", "OP13", "OP1", "OP5"]], id="Single row"),
    ],
)
def test_matrix_successful_initialization(data):
    # Test for successful initialization
    try:
        matrix = Matrix(data)
        assert isinstance(
            matrix, Matrix
        ), "Matrix should be successfully initialized with valid data."
    except ValueError:
        pytest.fail("Matrix initialization should not fail for valid data.")


@pytest.mark.parametrize(
    "matrix, readout_position, exp_error",
    [
        pytest.param(
            [["OP9", "OP13"]],
            {"OP9": "topleft", "OP13": "top-left"},
            r"Invalid readout position 'topleft' detected. Did you mean 'top-left'?",
            id="bad 'readout_position' 1",
        ),
        pytest.param(
            [["OP9"], ["OP13"]],
            {"OP9": "top-left", "OP13": "top-Right"},
            r"Invalid readout position 'top-Right' detected. Did you mean 'top-right'?",
            id="bad 'readout_position' 2",
        ),
        pytest.param(
            [["OP9"], ["OP13"]],
            {"OP9": "top-left", "OP13": "xxx"},
            r"Invalid readout position 'xxx' detected.",
            id="bad 'readout_position' 3",
        ),
    ],
)
def test_channels_bad_readout_position(matrix, readout_position, exp_error):
    """Test with a malformed 'readout_position' parameter."""
    with pytest.raises(ValueError, match=exp_error):
        _ = Channels(
            matrix=Matrix(matrix), readout_position=ReadoutPosition(readout_position)
        )


def test_eq():
    """Test method '__eq__'."""
    channels1_grid = Channels(
        matrix=Matrix([["OP9", "OP13"], ["OP1", "OP5"]]),  # Wrap list with Matrix
        readout_position=ReadoutPosition(
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",
                "OP5": "bottom-left",
            }
        ),
    )

    channels2_grid = Channels(
        matrix=Matrix([["OP13", "OP9"], ["OP1", "OP5"]]),  # Wrap list with Matrix
        readout_position=ReadoutPosition(
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",
                "OP5": "bottom-left",
            }
        ),
    )

    channels1_row = Channels(
        matrix=Matrix([["OP9", "OP13", "OP1", "OP5"]]),  # Wrap list with Matrix
        readout_position=ReadoutPosition(
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",
                "OP5": "bottom-left",
            }
        ),
    )

    channels1_column = Channels(
        matrix=Matrix([["OP13"], ["OP9"], ["OP1"], ["OP5"]]),  # Wrap list with Matrix
        readout_position=ReadoutPosition(
            {
                "OP9": "top-left",
                "OP13": "top-left",
                "OP1": "bottom-left",
                "OP5": "bottom-left",
            }
        ),
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
                matrix=Matrix([["OP9", "OP13"], ["OP1", "OP5"]]),
                readout_position=ReadoutPosition(
                    (
                        {
                            "OP9": "top-left",
                            "OP13": "top-left",
                            "OP1": "bottom-left",
                            "OP5": "bottom-left",
                        }
                    ),
                ),
            ),
            "Channels<4 channels>",
            id="Grid",
        ),
        pytest.param(
            Channels(
                matrix=Matrix([["OP9", "OP13"]]),
                readout_position=ReadoutPosition(
                    (
                        {
                            "OP9": "top-left",
                            "OP13": "top-left",
                        }
                    ),
                ),
            ),
            "Channels<2 channels>",
            id="Row",
        ),
        pytest.param(
            Channels(
                matrix=Matrix([["OP9"], ["OP1"]]),
                readout_position=ReadoutPosition(
                    (
                        {
                            "OP9": "top-left",
                            "OP1": "bottom-left",
                        }
                    ),
                ),
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
            [["OP9", "OP13"]],
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
    channels = Channels(
        matrix=Matrix(matrix), readout_position=ReadoutPosition(readout_position)
    )

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
                matrix=Matrix([["OP9", "OP13"], ["OP1", "OP5"]]),
                readout_position=ReadoutPosition(
                    (
                        {
                            "OP9": "top-left",
                            "OP13": "top-left",
                            "OP1": "bottom-left",
                            "OP5": "bottom-left",
                        }
                    ),
                ),
            ),
            id="Grid",
        ),
    ],
)
def test_from_dict(dct, exp_channels):
    """Test method '.from_dict'."""
    channels = Channels.from_dict(dct)

    assert channels == exp_channels
