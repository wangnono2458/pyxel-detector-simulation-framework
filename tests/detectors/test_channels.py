#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import pytest

from pyxel.detectors.channels import Channels
from pyxel.detectors.geometry import Geometry


def test_validate():
    channels = Channels(
        num_rows=514,
        num_cols=512,
        frame_mode="split",
        output={
            "channel_1": "right",
            "channel_2": "right",
            "channel_3": "right",
            "channel_4": "right",
        },
    )

    # Mock Geometry object with dimensions
    geometry = Geometry(row=1028, col=1024)

    channels.validate(geometry, full_frame_num_rows=1028, full_frame_num_cols=1024)


def test_validate_fail_geometry():
    # Initialize Channels with invalid num_rows and num_cols
    channels = Channels(
        num_rows=514,  # Not a divisor of 1024
        num_cols=512,
        frame_mode="split",
        output={
            "channel_1": "right",
            "channel_2": "right",
            "channel_3": "right",
            "channel_4": "right",
        },
    )

    # Mock Geometry object with dimensions
    geometry = Geometry(row=1024, col=1024)

    # Validate and expect a ValueError
    with pytest.raises(ValueError, match="'num_rows' .* must be a divisor of .*"):
        channels.validate(geometry, full_frame_num_rows=1024, full_frame_num_cols=1024)


def test_validate_fail_frame():

    with pytest.raises(
        ValueError, match="'frame_mode' must be one of 'top', 'bottom', or 'split'."
    ):
        Channels(
            num_rows=514,
            num_cols=512,
            frame_mode="center",
            output={
                "channel_1": "right",
                "channel_2": "right",
                "channel_3": "right",
                "channel_4": "right",
            },
        )


def test_validate_fail_negative_rows_cols():

    with pytest.raises(ValueError, match="'num_rows' must be non-negative."):
        Channels(num_rows=-1, num_cols=10, frame_mode="top", output={})

    with pytest.raises(ValueError, match="'num_cols' must be non-negative."):
        Channels(num_rows=10, num_cols=-5, frame_mode="top", output={})


def test_invalid_output_value():
    # Attempt to create a Channels instance with an invalid output direction
    with pytest.raises(ValueError, match="must be either 'left' or 'right'"):
        Channels(
            num_rows=4,
            num_cols=4,
            frame_mode="split",
            output={
                "channel_1": "left",
                "channel_2": "up",  # Invalid direction
            },
        )


def test_validate_output_count_valid():
    # Test with valid output count
    channels = Channels(
        num_rows=512,
        num_cols=512,
        frame_mode="split",
        output={
            "channel_1": "left",
            "channel_2": "right",
            "channel_3": "left",
            "channel_4": "right",
        },
    )
    geometry = Geometry(row=1024, col=1024)
    # This should pass as 1024/512 * 1024/512 = 4
    channels.validate(geometry, full_frame_num_rows=1024, full_frame_num_cols=1024)


def test_validate_output_count_invalid():
    # Test with invalid output count
    invalid_channels = Channels(
        num_rows=512,
        num_cols=512,
        frame_mode="split",
        output={
            "channel_1": "left",
            "channel_2": "right",
        },  # Only 2 outputs instead of 4
    )
    geometry = Geometry(row=1024, col=1024)

    with pytest.raises(ValueError, match="must match the number of outputs provided"):
        invalid_channels.validate(
            geometry, full_frame_num_rows=1024, full_frame_num_cols=1024
        )
