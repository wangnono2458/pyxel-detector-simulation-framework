#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from typing_extensions import Self


class Matrix:
    """Class to store and validate the matrix structure."""

    def __init__(self, data: Sequence[Sequence[str]]):
        if not all(isinstance(row, list) for row in data):
            raise ValueError("Matrix must be a list of lists.")

        self.data = data

    def flatten(self):
        """Flatten the matrix into a single list of terms."""
        return [term for row in self.data for term in row]

    def __len__(self):
        """Return the total number of elements in the matrix."""
        return sum(len(row) for row in self.data)


class ReadoutPosition:
    """Class to store and validate readout positions."""

    VALID_POSITIONS = {"top-left", "top-right", "bottom-left", "bottom-right"}

    def __init__(self, positions: Mapping[str | int, str]):
        if not all(pos in self.VALID_POSITIONS for pos in positions.values()):
            raise ValueError("Invalid readout position detected.")

        self.positions = positions

    def keys(self):
        """Return the keys of the readout positions."""
        return set(self.positions.keys())

    def __len__(self):
        """Return the number of readout positions."""
        return len(self.positions)


class Channels:
    """Updated Channels class using Matrix and ReadoutPosition."""

    def __init__(
        self,
        matrix: Sequence[Sequence[str]],
        readout_position: Mapping[
            str | int, Literal["top-left", "top-right", "bottom-left", "bottom-right"]
        ],
    ):
        self.matrix = Matrix(matrix)
        self.readout_position = ReadoutPosition(readout_position)

        # Validate matching counts
        if len(self.matrix) > len(self.readout_position):
            raise ValueError("Readout direction of at least one channel is missing.")

        # Validate that matrix terms exist in readout positions
        matrix_terms = set(self.matrix.flatten())
        readout_keys = self.readout_position.keys()

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

    #
    # def validate(
    #     self, geometry,
    #         # full_frame_num_rows: int, full_frame_num_cols: int
    # ) -> None:
    #     """
    #     Validate that num_rows and num_cols are divisors of geometry's dimensions.
    #
    #     :param geometry: An instance of the Geometry class.
    #     :raises ValueError: If row or col are not divisors of geometry's dimensions.
    #
    #     Parameters
    #     ----------
    #     full_frame_num_cols
    #     full_frame_num_rows
    #     """
    #
    #     full_frame_num_rows = geometry.row
    #     full_frame_num_cols = geometry.col
    #
    #     if full_frame_num_rows % self.num_rows != 0:
    #         raise ValueError(
    #             f"'num_rows' ({self.num_rows}) must be a divisor of full_frame_num_rows ({full_frame_num_rows})."
    #         )
    #     if full_frame_num_cols % self.num_cols != 0:
    #         raise ValueError(
    #             f"'num_cols' ({self.num_cols}) must be a divisor of full_frame_num_cols ({full_frame_num_cols})."
    #         )
    #     if (full_frame_num_rows % self.num_rows == 0) and (
    #         full_frame_num_cols % self.num_cols == 0
    #     ):
    #         # Validate that the product of the divisions equals the number of outputs
    #         rows_division = full_frame_num_rows // self.num_rows
    #         cols_division = full_frame_num_cols // self.num_cols
    #         expected_output_count = rows_division * cols_division
    #
    #         if len(self.output) != expected_output_count:
    #             raise ValueError(
    #                 f"The product of the divisions ({expected_output_count}) must match the number of outputs provided ({len(self.output)})."
    #             )

    def get_channel_coord(self, channel) -> tuple[slice, slice]:
        raise NotImplementedError

    def build_mask(self) -> np.ndarray:
        # Should save n array, one for each channel? Or should it have a well-defined structure to identify the channels?
        raise NotImplementedError

    # TODO:Convert matrix into numpy array

    # @property
    # def num_rows(self) -> float:
    #     """Get number of rows of the channels."""
    #     if self.num_rows is None:
    #         raise ValueError("'num rows' not specified in detector environment.")
    #
    #     return self.num_rows
    #
    # @num_rows.setter
    # def num_rows(self, value: int | float) -> None:
    #     """Set number of rows of the detector."""
    #     if isinstance(value, (int, float)):
    #         if value <= 0.0:
    #             raise ValueError("'num rows' must be strictly positive.")
    #     elif not isinstance(value):
    #         raise TypeError("A NumHandling object or a float must be provided.")

    def to_dict(self) -> Mapping:
        """Get the attributes of this instance as a `dict`."""
        return {"matrix": self.matrix, "readout_position": self.readout_position}

    #
    # @classmethod
    # def from_dict(cls, dct: Mapping) -> Self:
    #     """Create a new instance of `Geometry` from a `dict`."""
    #     return cls(**dct)
    #     value = dct.get("num rows")
    #
    #     if value is None:
    #         num_rows: float | None = None
    #     elif isinstance(value, (int, float)):
    #         num_rows = float(value)
    #     # elif isinstance(value, dict):
    #     #    num_rows = NumHandling.from_dict(value)
    #     else:
    #         raise NotImplementedError
    #
    #     return cls(num_rows=num_rows)

    @classmethod
    def from_dict(cls, dct: Mapping) -> Self:
        """Create a new instance of `Geometry` from a `dict`."""
        # TODO: This is a simplistic implementation. Improve this.
        obj = cls(**dct)
        return obj
