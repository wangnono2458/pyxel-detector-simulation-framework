# Copyright (c) 2023 Arthur Kadela and Joonas Viuho, Niels Bohr Institute, University of Copenhagen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pytest

from pyxel.detectors import CCD
from pyxel.models.charge_transfer import multiplication_register_cic


def test_multiplication_register_cic(ccd_10x10: CCD):
    detector = ccd_10x10
    detector.pixel.non_volatile.array = np.zeros(detector.geometry.shape, dtype=float)

    multiplication_register_cic(
        detector=detector, total_gain=0, gain_elements=1, pcic_rate=0.0, scic_rate=0.0
    )


@pytest.mark.parametrize(
    "total_gain,gain_elements,pcic_rate,scic_rate",
    [
        (-1, 10, 0.0, 0.0),
        (1, -1, -1, 1),
        (-1, -1, 0.005, -0.02),
    ],
)
def test_multiplication_register_cic_bad_inputs(
    ccd_10x10: CCD, total_gain, gain_elements, pcic_rate, scic_rate
):
    detector = ccd_10x10

    with pytest.raises(ValueError, match="Wrong input parameter"):
        multiplication_register_cic(
            detector=detector,
            total_gain=total_gain,
            gain_elements=gain_elements,
            pcic_rate=pcic_rate,
            scic_rate=scic_rate,
        )
