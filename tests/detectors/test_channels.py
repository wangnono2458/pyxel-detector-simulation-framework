#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from collections.abc import Mapping, Sequence

import pytest

from pyxel.detectors.channels import Channels
from pyxel.detectors.geometry import Geometry


def test_channels_valid_initialization():
    """Test Channels initialization with correct matrix and readout_position."""
    matrix = [["OP9", "OP13"], ["OP1", "OP5"]]
    readout_position = {
        "OP9": "top-left",
        "OP13": "top-left",
        "OP1": "bottom-left",
        "OP5": "bottom-left",
    }

    # Should not raise an error
    channels = Channels(matrix=matrix, readout_position=readout_position)

    # Check if attributes are correctly assigned
    assert channels.matrix.data == matrix
    assert channels.readout_position.positions == readout_position


def test_channels_missing_readout_position():
    """Test Channels raises an error when readout_position is incomplete."""
    matrix = [["OP9", "OP13"], ["OP1", "OP5"]]
    readout_position = {
        "OP9": "top-left",
        "OP13": "top-left",
        # Missing OP1 and OP5
    }

    with pytest.raises(
        ValueError, match="Readout direction of at least one channel is missing."
    ):
        Channels(matrix=matrix, readout_position=readout_position)


def test_channels_mismatched_matrix_and_readout():
    """Test Channels raises an error when a channel in the matrix is not in readout_position."""
    matrix = [["OP9", "OP13"], ["OP1", "OP5"]]
    readout_position = {
        "OP9": "top-left",
        "OP13": "top-left",
        "OP1": "bottom-left",
        # OP5 is missing
    }

    with pytest.raises(
        ValueError, match="Readout direction of at least one channel is missing."
    ):
        Channels(matrix=matrix, readout_position=readout_position)


def test_channels_extra_readout_position():
    """Test Channels raises an error when readout_position has extra keys."""
    matrix = [["OP9", "OP13"], ["OP1", "OP5"]]
    readout_position = {
        "OP9": "top-left",
        "OP13": "top-left",
        "OP1": "bottom-left",
        "OP5": "bottom-left",
        "OP_EXTRA": "bottom-right",  # Extra key that isn't in matrix
    }

    with pytest.raises(
        ValueError,
        match="Readout position contains extra channels not listed in matrix.",
    ):
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
