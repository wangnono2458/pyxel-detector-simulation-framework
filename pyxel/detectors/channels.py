#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Sub-package to handle and validate matrix structures and readout positions for multi-channels detectors.

**Example of four channels**

In this example four channels ``OP9``, ``OP13``, ``OP1`` and ``OP5`` are
defined in a matrix configuration as follows:

.. figure:: _static/channels.png
    :scale: 70%
    :alt: Channels
    :align: center

The corresponding YAML definition could be:

.. code-block:: yaml


 geometry:
    row: 1028
    col: 1024
    channels:
      matrix: [[OP9, OP13],
               [OP1, OP5 ]]
      readout_position:
        - OP9:  top-left
        - OP13: top-left
        - OP1:  bottom-left
        - OP5:  bottom-left
"""

import difflib
from collections.abc import Hashable, Mapping, Sequence
from typing import Literal

import numpy as np
from typing_extensions import Self


class Matrix:
    """Class to store and validate the 1D or 2D matrix structure.

    Examples
    --------
    >>> matrix = Matrix([["OP9", "OP13"], ["OP1", "OP5"]])
    >>> matrix.shape
    (2, 2)
    >>> matrix.size
    4
    """

    def __init__(self, data: Sequence[Sequence[Hashable]]):
        # First check to ensure matrix is not empty
        if not data:
            raise ValueError("Matrix data must contain at least one row.")

        # Validate that data is a sequence of sequences and not a string or a sequence of strings
        if not isinstance(data, Sequence) or isinstance(data, str):
            raise TypeError(
                "Matrix must be a sequence of sequences (e.g., list of lists)."
            )

        # Validate that each item in the data is also a sequence and not a string
        if any(isinstance(row, str) or not isinstance(row, Sequence) for row in data):
            raise ValueError(
                "All rows in the matrix must be sequences and not strings."
            )

        # Ensure all rows are of the same length
        row_lengths = [len(row) for row in data]
        if len(set(row_lengths)) != 1:
            raise ValueError(
                "Parameter 'matrix' is malformed: All rows must be of the same length."
            )

        # Check if rows are empty and the matrix contains more than one row
        if any(len(row) == 0 for row in data) and len(data) > 1:
            raise ValueError("Parameter 'matrix' is malformed: Cannot have empty rows.")

        try:
            self._data = np.array(
                data, dtype=object
            )  # Use dtype=object to accommodate different data types
        except Exception as exc:
            raise ValueError("Failed to convert input data to a NumPy array: ") from exc

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and np.array_equal(self._data, other._data)

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        return np.array(self._data, dtype=dtype)

    @property
    def size(self) -> int:
        """Return number of elements in the matrix."""
        return self._data.size

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the matrix."""
        return self._data.shape  # type: ignore[return-value]


# TODO: Implement using Abstract Class 'Mapping'
class ReadoutPosition:
    """Class to store and validate the mapping between channel names and their physical readout positions.

    Examples
    --------
    >>> positions = ReadoutPosition(
    ...     positions={
    ...         "OP9": "top-left",
    ...         "OP13": "top-left",
    ...         "OP1": "bottom-left",
    ...         "OP5": "bottom-left",
    ...     }
    ... )

    >>> len(positions)
    4

    >>> list(positions)
    ['OP9', 'OP13', 'OP1', 'OP5']

    >>> positions["OP9"]
    'top_left'
    """

    # TODO: Refactor using a 'TypeAlias'
    VALID_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")

    def __init__(
        self,
        positions: Mapping[
            str,
            Literal[
                "top-left", "top-right", "bottom-left", "bottom-right"
            ],  # TODO: Refactor using a 'TypeAlias'
        ],
    ):
        # Validate that all provided 'positions' are correct
        if not set(positions.values()).issubset(self.VALID_POSITIONS):
            wrong_positions = set(positions.values()).difference(self.VALID_POSITIONS)

            first_wrong_position, *_ = list(wrong_positions)
            first_close_match_position, *_ = difflib.get_close_matches(
                word=first_wrong_position,
                possibilities=self.VALID_POSITIONS,
            )

            raise ValueError(
                f"Invalid readout position {first_wrong_position!r} detected. "
                f"Did you mean {first_close_match_position!r}?"
            )

        self.positions = positions

    def __eq__(self, other):
        if not isinstance(other, ReadoutPosition):
            return NotImplemented
        return self.positions == other.positions

    def __len__(self):
        """Return the number of readout positions."""
        return len(self.positions)

    def keys(self):
        """Return the keys of the readout positions."""
        return set(self.positions.keys())


class Channels:
    """Updated Channels class with their matrix layout and corresponding readout positions.

    Examples
    --------
    >>> channels = Channels(
    ...     matrix=Matrix([["OP9", "OP13"], ["OP1", "OP5"]]),
    ...     readout_position=ReadoutPosition(
    ...         {
    ...             "OP9": "top-left",
    ...             "OP13": "top-left",
    ...             "OP1": "bottom-left",
    ...             "OP5": "bottom-left",
    ...         }
    ...     ),
    ... )
    >>> channels
    Channels<4 channels>
    """

    def __init__(self, matrix: Matrix, readout_position: ReadoutPosition):
        self.matrix = matrix
        self.readout_position = readout_position

        # Validate matching counts
        # Flatten the matrix to count unique elements
        matrix_terms = {term for row in self.matrix._data for term in row}
        readout_keys = set(self.readout_position.keys())

        if len(matrix_terms) > len(readout_keys):
            raise ValueError("Readout direction of at least one channel is missing.")

        # Ensure all matrix terms exist in readout positions
        if not matrix_terms.issubset(readout_keys):
            raise ValueError(
                "Channel names in the matrix and in the readout directions are not matching."
            )

        # Ensure no extra channels in readout_position
        extra_keys = readout_keys - matrix_terms
        if extra_keys:
            raise ValueError(
                "Readout position contains extra channels not listed in matrix."
            )

    def __eq__(self, other):
        if not isinstance(other, Channels):
            return NotImplemented

        # Check equality by comparing both the matrix and the readout positions
        return (
            self.matrix == other.matrix
            and self.readout_position == other.readout_position
        )

    def __repr__(self):
        # Count unique channel names by flattening the matrix and getting unique items
        unique_channels = {term for row in self.matrix._data for term in row}
        return f"Channels<{len(unique_channels)} channels>"

    # TODO: Move
    def build_mask(self) -> np.ndarray:
        """Generate a mask or map for the defined channels."""
        # Should save n array, one for each channel? Or should it have a well-defined structure to identify the channels?
        raise NotImplementedError

    def to_dict(self) -> Mapping:
        """Get the attributes of this instance as a `dict`."""
        return {
            "matrix": np.asarray(self.matrix).tolist(),
            "readout_position": dict(self.readout_position.positions),
        }

    @classmethod
    def from_dict(cls, dct: Mapping) -> Self:
        """Create a new instance of `Geometry` from a `dict`."""
        # TODO: This is a simplistic implementation. Improve this.

        obj = cls(
            matrix=Matrix(dct["matrix"]),
            readout_position=ReadoutPosition(dct["readout_position"]),
        )
        return obj
