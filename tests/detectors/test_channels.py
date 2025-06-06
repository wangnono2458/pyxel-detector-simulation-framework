#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

from pyxel.detectors.channels import Channels, Matrix, ReadoutPosition


@dataclass
class MatrixParams:
    matrix: list
    shape: tuple[int, ...]
    ndim: int
    flat_list: list


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
            MatrixParams(
                matrix=[["OP9", "OP13"], ["OP1", "OP5"]],
                shape=(2, 2),
                ndim=2,
                flat_list=["OP9", "OP13", "OP1", "OP5"],
            ),
            id="Grid structure (str)",
        ),
        pytest.param(
            [[9, 13], [1, 5]],
            {9: "top-left", 13: "top-left", 1: "bottom-left", 5: "bottom-left"},
            MatrixParams(
                matrix=[[9, 13], [1, 5]], shape=(2, 2), ndim=2, flat_list=[9, 13, 1, 5]
            ),
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
            MatrixParams(
                matrix=[["OP9", 13], [1, "OP5"]],
                shape=(2, 2),
                ndim=2,
                flat_list=["OP9", 13, 1, "OP5"],
            ),
            id="Grid structure (str and int)",
        ),
        pytest.param(
            [["OP9", "OP13"]],
            {"OP9": "top-left", "OP13": "top-left"},
            MatrixParams(
                matrix=[["OP9", "OP13"]],
                shape=(1, 2),
                ndim=2,
                flat_list=["OP9", "OP13"],
            ),
            id="row structure",
        ),
        pytest.param(
            [["OP9"], ["OP1"]],
            {"OP9": "top-left", "OP1": "bottom-left"},
            MatrixParams(
                matrix=[["OP9"], ["OP1"]],
                shape=(2, 1),
                ndim=2,
                flat_list=["OP9", "OP1"],
            ),
            id="column structure",
        ),
    ],
)
def test_channels_valid_initialization(
    matrix, readout_position, exp_matrix: MatrixParams
):
    """Test Channels initialization with correct matrix and readout_position."""
    # Should not raise an error
    channels = Channels(
        matrix=Matrix(matrix),
        readout_position=ReadoutPosition(readout_position),
    )

    # Check if attributes are correctly assigned
    assert isinstance(channels.matrix, Matrix)
    assert np.array(channels.matrix).tolist() == exp_matrix.matrix
    assert channels.shape == exp_matrix.shape
    assert channels.ndim == exp_matrix.ndim
    assert list(channels) == exp_matrix.flat_list

    assert channels.readout_position.positions == readout_position


@pytest.mark.parametrize(
    "obj1, obj2,exp_result",
    [
        pytest.param(
            Channels(
                matrix=Matrix([["OP9", "OP13"]]),
                readout_position=ReadoutPosition(
                    {"OP9": "top-left", "OP13": "top-left"}
                ),
            ),
            Channels(
                matrix=Matrix([["OP9", "OP13"]]),
                readout_position=ReadoutPosition(
                    {"OP9": "top-left", "OP13": "top-left"}
                ),
            ),
            True,
            id="same",
        ),
        pytest.param(
            Channels(
                matrix=Matrix([["OP9", "OP13"]]),
                readout_position=ReadoutPosition(
                    {"OP9": "top-left", "OP13": "top-left"}
                ),
            ),
            Channels(
                matrix=Matrix([["OP9", "OP13"]]),
                readout_position=ReadoutPosition(
                    {
                        "OP13": "top-left",
                        "OP9": "top-left",
                    }
                ),
            ),
            True,
            id="same2",
        ),
        pytest.param(
            Channels(
                matrix=Matrix([["OP9", "OP13"]]),
                readout_position=ReadoutPosition(
                    {"OP9": "top-left", "OP13": "top-left"}
                ),
            ),
            Channels(
                matrix=Matrix([["OP9"], ["OP13"]]),
                readout_position=ReadoutPosition(
                    {"OP9": "top-left", "OP13": "top-left"}
                ),
            ),
            False,
            id="different shape",
        ),
        pytest.param(
            Channels(
                matrix=Matrix([["OP9", "OP13"]]),
                readout_position=ReadoutPosition(
                    {"OP9": "top-left", "OP13": "top-left"}
                ),
            ),
            Channels(
                matrix=Matrix([["OP9", "OP13"]]),
                readout_position=ReadoutPosition(
                    {"OP9": "top-left", "OP13": "top-right"}
                ),
            ),
            False,
            id="different readout positions",
        ),
        pytest.param(
            Channels(
                matrix=Matrix([["OP9", "OP13"]]),
                readout_position=ReadoutPosition(
                    {"OP9": "top-left", "OP13": "top-left"}
                ),
            ),
            Channels(
                matrix=Matrix([["OP13", "OP9"]]),
                readout_position=ReadoutPosition(
                    {"OP9": "top-left", "OP13": "top-left"}
                ),
            ),
            False,
            id="different matrix",
        ),
        pytest.param(
            Channels(
                matrix=Matrix([["OP9", "OP13"]]),
                readout_position=ReadoutPosition(
                    {"OP9": "top-left", "OP13": "top-left"}
                ),
            ),
            Matrix([["OP9", "OP13"]]),
            False,
            id="different type",
        ),
    ],
)
def test_channels_eq(obj1, obj2, exp_result: bool):
    """Test method 'Channels.__eq__'."""
    assert isinstance(obj1, Channels)

    if exp_result:
        assert obj1 == obj2
    else:
        assert obj1 != obj2


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
            matrix=Matrix(matrix),
            readout_position=ReadoutPosition(readout_position),
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
            matrix=Matrix(matrix),
            readout_position=ReadoutPosition(readout_position),
        )


@pytest.mark.parametrize(
    "matrix, readout_position, exp_message",
    [
        pytest.param(
            [["OP9", "OP13"]],
            ReadoutPosition({"OP9": "top-left", "OP13": "top-left"}),
            r"'matrix' must be a Matrix",
            id="Wrong 'matrix'",
        ),
        pytest.param(
            Matrix([["OP9", "OP13"]]),
            {"OP9": "top-left", "OP13": "top-left"},
            r"'readout_position' must be a ReadoutPosition",
            id="Wrong 'readout_position'",
        ),
    ],
)
def test_channels_wrong_initialization(matrix, readout_position, exp_message: str):
    """Test 'Channels.__init__' with wrong parameters."""
    with pytest.raises(TypeError, match=exp_message):
        _ = Channels(matrix=matrix, readout_position=readout_position)


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
            matrix=Matrix(matrix),
            readout_position=ReadoutPosition(readout_position),
        )


@pytest.mark.parametrize(
    "data, exp_exc, exp_message",
    [
        pytest.param(
            [["OP9"], ["OP13", "OP1", "OP5"]],
            ValueError,
            "Parameter 'matrix' is malformed: All rows must be of the same length.",
            id="Mismatched row lengths",
        ),
        pytest.param(
            [["OP9"], ["OP13", "OP1"], ["OP5"]],
            ValueError,
            "Parameter 'matrix' is malformed: All rows must be of the same length.",
            id="Each row different lengths",
        ),
        pytest.param(
            ["OP9", "OP13", "OP1", "OP5"],
            ValueError,
            "All rows in the matrix must be sequences and not strings.",
            id="Single string inputs treated as rows",
        ),
        pytest.param(
            [],
            ValueError,
            "Matrix data must contain at least one row.",
            id="Empty matrix",
        ),
        pytest.param(
            [[9, 13], [1]],
            ValueError,
            "Parameter 'matrix' is malformed: All rows must be of the same length.",
            id="Numerical row length mismatch",
        ),
        pytest.param(
            [[], []],
            ValueError,
            "Parameter 'matrix' is malformed: Cannot have empty rows.",
            id="Empty rows but equal length",
        ),
        pytest.param("OP9", TypeError, "Matrix must be a sequence", id="string"),
    ],
)
def test_matrix_initialization(data, exp_exc, exp_message):
    if exp_message:
        with pytest.raises(exp_exc, match=exp_message):
            _ = Matrix(data)
    else:
        # Expecting successful initialization here
        matrix = Matrix(data)
        assert isinstance(matrix, Matrix)


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
    matrix = Matrix(data)
    assert isinstance(matrix, Matrix)


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
            matrix=Matrix(matrix),
            readout_position=ReadoutPosition(readout_position),
        )


@pytest.mark.parametrize(
    "obj1, obj2, exp_result",
    [
        pytest.param(
            ReadoutPosition({"OP9": "top-left", "OP13": "top-right"}),
            ReadoutPosition({"OP9": "top-left", "OP13": "top-right"}),
            True,
            id="same",
        ),
        pytest.param(
            ReadoutPosition({"OP9": "top-left", "OP13": "top-right"}),
            ReadoutPosition({"OP13": "top-right", "OP9": "top-left"}),
            True,
            id="same reversed",
        ),
        pytest.param(
            ReadoutPosition({"OP9": "top-left", "OP13": "top-right"}),
            ReadoutPosition({"OP9": "top-left", "OP13": "top-left"}),
            False,
            id="different",
        ),
        pytest.param(
            ReadoutPosition({"OP9": "top-left", "OP13": "top-right"}),
            {"OP9": "top-left", "OP13": "top-right"},
            False,
            id="different type",
        ),
    ],
)
def test_readout_positions_eq(obj1, obj2, exp_result: bool):
    """Test method 'ReadoutPosition.__eq__'."""
    assert isinstance(obj1, ReadoutPosition)

    if exp_result:
        assert obj1 == obj2
    else:
        assert obj1 != obj2


@pytest.mark.parametrize(
    "position, exp_len",
    [
        pytest.param(ReadoutPosition({"OP9": "top-left"}), 1, id="1 element"),
        pytest.param(
            ReadoutPosition({"OP9": "top-left", "OP13": "top-right"}),
            2,
            id="2 elements",
        ),
    ],
)
def test_readout_positions_len(position, exp_len: int):
    """Test method 'ReadoutPosition.__len__'."""
    assert isinstance(position, ReadoutPosition)
    assert len(position) == exp_len


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
def test_channels_from_dict(dct, exp_channels):
    """Test method 'Channels.from_dict'."""
    channels = Channels.from_dict(dct)

    assert channels == exp_channels


@pytest.mark.parametrize(
    "dct, exp_exc, exp_msg",
    [
        pytest.param(
            {"readout_position": {"OP9": "top-left", "OP13": "top-left"}},
            KeyError,
            r"Missing required key 'matrix'",
        ),
        pytest.param(
            {"matrix": [["OP9", "OP13"]]},
            KeyError,
            r"Missing required key 'readout_position'",
        ),
    ],
)
def test_channels_from_dict_bad_inputs(dct, exp_exc, exp_msg: str):
    """Test method 'Channels.from_dict' with bad inputs."""
    with pytest.raises(exp_exc, match=exp_msg):
        _ = Channels.from_dict(dct)
