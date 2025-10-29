#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
import numpy as np
import pytest
import xarray as xr
from skimage.draw import line_aa

from pyxel.detectors import (
    CCD,
    CCDGeometry,
    Characteristics,
    Environment,
    ReadoutProperties,
)
from pyxel.models.data_processing import remove_cosmic_rays

# Check if 'lacosmic' is installed
_ = pytest.importorskip(
    "lacosmic",
    reason="Package 'lacosmic' is not installed. Use 'pip install lacosmic'",
)


def create_image_with_cosmics():
    data = np.zeros(shape=(50, 50), dtype=float)

    # rr, cc = disk((10, 30), 3, shape=None)
    # data[rr, cc] = 200.0
    #
    # rr, cc = disk((30, 10), 5, shape=None)
    # data[rr, cc] = 200.0
    #
    # # data[25, :] = 500.0
    # data[24, :] = 500.0

    rr, cc, val = line_aa(1, 1, 8, 8)
    data[rr, cc] = val * 500.0

    rr, cc, val = line_aa(1, 28, 8, 21)
    data[rr, cc] = val * 300.0

    rr, cc, val = line_aa(25, 25, 26, 26)
    data[rr, cc] = val * 300.0

    # rr, cc, val = circle_perimeter_aa(30, 30, 15)
    # data[rr, cc] = val * 200.0
    #
    rr, cc, val = line_aa(20, 10, 20, 30)
    data[rr, cc] = val * 100.0

    return data


@pytest.fixture
def ccd_50x50() -> CCD:
    """Create a valid CCD detector."""
    detector = CCD(
        geometry=CCDGeometry(
            row=50,
            col=50,
            total_thickness=40.0,
            pixel_vert_size=10.0,
            pixel_horz_size=10.0,
        ),
        environment=Environment(),
        characteristics=Characteristics(),
    )
    detector._readout_properties = ReadoutProperties(times=[1.0])
    return detector


def test_data_output(ccd_50x50: CCD):
    """Test output for function 'remove_cosmic_rays'."""
    detector = ccd_50x50
    pixels = create_image_with_cosmics()  # 2d array
    detector.pixel.non_volatile.array = pixels
    remove_cosmic_rays(detector=detector)
    data = detector.data

    assert isinstance(data, xr.DataTree)
    assert "lacosmic" in data

    assert isinstance(data["/lacosmic/cosmic_ray_clean"], xr.DataArray)
    assert isinstance(data["/lacosmic/cosmic_ray_mask"], xr.DataArray)

    assert list(data["/lacosmic/cosmic_ray_clean"].coords.keys()) == ["y", "x", "time"]
    assert list(data["/lacosmic/cosmic_ray_mask"].coords.keys()) == ["y", "x", "time"]

    assert list(data["/lacosmic/cosmic_ray_clean"].coords["time"]) == [0.0]
    assert list(data["/lacosmic/cosmic_ray_mask"].coords["time"]) == [0.0]


def test_ray_removal(ccd_50x50: CCD):
    """Test ray tracing removal."""
    detector = ccd_50x50

    pixels = create_image_with_cosmics()  # 2d array
    detector.pixel.non_volatile.array = pixels

    ###################
    # First iteration #
    ###################
    detector.readout_properties.time = 0.0
    detector.readout_properties.time_step = 1.0
    detector.readout_properties.pipeline_count = 0

    remove_cosmic_rays(detector=detector)

    ####################
    # Second iteration #
    ####################
    detector.readout_properties.time = 1.0
    detector.readout_properties.time_step = 1.0
    detector.readout_properties.pipeline_count = 1

    remove_cosmic_rays(detector=detector)

    data = detector.data

    assert np.sum(data["/lacosmic/cosmic_ray_mask"].data) > 0
    assert np.sum(data["/lacosmic/cosmic_ray_clean"].data) == 0.0
