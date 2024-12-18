#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

from collections.abc import Mapping
from typing import Literal

import numpy as np
from typing_extensions import Self


class Channels:
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        frame_mode: Literal["top", "bottom", "split"],
        output: Mapping[str, Literal["left", "right"]],
    ):
        # Validate frame_mode
        if frame_mode not in ("top", "bottom", "split"):
            raise ValueError("'frame_mode' must be one of 'top', 'bottom', or 'split'.")

        # Validate num_rows and num_cols are non-negative
        if num_rows < 0:
            raise ValueError("'num_rows' must be non-negative.")
        if num_cols < 0:
            raise ValueError("'num_cols' must be non-negative.")

        # Validate output dictionary values
        for channel, direction in output.items():
            if direction not in ("left", "right"):
                raise ValueError(
                    f"The output direction for '{channel}' must be either 'left' or 'right', not '{direction}'."
                )

        self.num_rows: int = num_rows
        self.num_cols: int = num_cols
        self.frame_mode: Literal["top", "bottom", "split"] = frame_mode
        self.output = output

    def validate(
        self, geometry, full_frame_num_rows: int, full_frame_num_cols: int
    ) -> None:
        """
        Validate that num_rows and num_cols are divisors of geometry's dimensions.

        :param geometry: An instance of the Geometry class.
        :raises ValueError: If row or col are not divisors of geometry's dimensions.

        Parameters
        ----------
        full_frame_num_cols
        full_frame_num_rows
        """

        full_frame_num_rows = geometry.row
        full_frame_num_cols = geometry.col

        if full_frame_num_rows % self.num_rows != 0:
            raise ValueError(
                f"'num_rows' ({self.num_rows}) must be a divisor of full_frame_num_rows ({full_frame_num_rows})."
            )
        if full_frame_num_cols % self.num_cols != 0:
            raise ValueError(
                f"'num_cols' ({self.num_cols}) must be a divisor of full_frame_num_cols ({full_frame_num_cols})."
            )
        if (full_frame_num_rows % self.num_rows == 0) and (
            full_frame_num_cols % self.num_cols == 0
        ):
            # Validate that the product of the divisions equals the number of outputs
            rows_division = full_frame_num_rows // self.num_rows
            cols_division = full_frame_num_cols // self.num_cols
            expected_output_count = rows_division * cols_division

            if len(self.output) != expected_output_count:
                raise ValueError(
                    f"The product of the divisions ({expected_output_count}) must match the number of outputs provided ({len(self.output)})."
                )

    def get_channel_coord(self, channel) -> tuple[slice, slice]:
        raise NotImplementedError

    def build_mask(self) -> np.ndarray:
        # Should save n array, one for each channel? Or should it have a well-defined structure to identify the channels?
        raise NotImplementedError

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
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "frame_mode": self.frame_mode,
            "output": self.output,
        }

    # @classmethod
    # def from_dict(cls, dct: Mapping) -> Self:
    #     """Create a new instance of `Geometry` from a `dict`."""
    #
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
    def from_dict(cls, dct: Mapping):
        """Create a new instance of `Geometry` from a `dict`."""
        # TODO: This is a simplistic implementation. Improve this.
        return cls(**dct)
