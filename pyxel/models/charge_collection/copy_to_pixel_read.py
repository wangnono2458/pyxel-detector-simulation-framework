#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.


"""PixelRead collection model."""

from pyxel.detectors import Detector
import numpy as np


def copy_to_pixel_read(
    detector: Detector,
) -> None:
    """Copy `pixel` array to `pixel_read` array.
    """

    detector.pixel_read.array = np.copy(detector.pixel)
