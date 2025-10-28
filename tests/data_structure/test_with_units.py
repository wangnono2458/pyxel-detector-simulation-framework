#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import numpy as np
import pytest
from astropy.units import Quantity

from pyxel.detectors import CCD, CCDGeometry, Characteristics, Environment


@pytest.fixture
def empty_detector() -> CCD:
    ccd = CCD(
        geometry=CCDGeometry(row=2, col=2),
        environment=Environment(),
        characteristics=Characteristics(),
    )

    return ccd


@pytest.mark.parametrize("with_unit", [True, False])
def test_with_photon(empty_detector: CCD, with_unit: bool):
    """Test assignment with Photon."""
    detector = empty_detector
    data_2d = np.array([[1, 2], [3, 4]], dtype=float)

    if with_unit:
        detector.photon = Quantity(data_2d, unit="ph")
        detector.photon += Quantity(data_2d, unit="photon")
        detector.photon = detector.photon + Quantity(data_2d, unit="ph")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.photon = Quantity(data_2d, unit="electron")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.photon += Quantity(data_2d, unit="electron")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.photon = detector.photon + Quantity(data_2d, unit="electron")

    else:
        detector.photon = data_2d
        detector.photon += data_2d
        detector.photon = detector.photon + data_2d

    np.testing.assert_equal(data_2d, np.array([[1, 2], [3, 4]], dtype=float))
    np.testing.assert_equal(detector.photon, 3 * data_2d)

    result = Quantity(detector.photon)
    np.testing.assert_equal(result.value, 3 * data_2d)
    assert result.unit == "ph"


@pytest.mark.parametrize("with_unit", [True, False])
def test_with_pixel(empty_detector: CCD, with_unit: bool):
    """Test assignment with Pixel."""
    detector = empty_detector
    data_2d = np.array([[1, 2], [3, 4]], dtype=float)

    if with_unit:
        detector.pixel.non_volatile = Quantity(data_2d, unit="electron")
        detector.pixel.non_volatile += Quantity(data_2d, unit="electron")
        detector.pixel.non_volatile = detector.pixel.non_volatile + Quantity(
            data_2d, unit="electron"
        )

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.pixel.non_volatile = Quantity(data_2d, unit="volt")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.pixel.non_volatile += Quantity(data_2d, unit="volt")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.pixel.non_volatile = detector.pixel.non_volatile + Quantity(
                data_2d, unit="volt"
            )

    else:
        detector.pixel.non_volatile = data_2d
        detector.pixel.non_volatile += data_2d
        detector.pixel.non_volatile = detector.pixel + data_2d

    np.testing.assert_equal(data_2d, np.array([[1, 2], [3, 4]], dtype=float))
    np.testing.assert_equal(detector.pixel, 3 * data_2d)

    result = Quantity(detector.pixel)
    np.testing.assert_equal(result.value, 3 * data_2d)
    assert result.unit == "electron"


@pytest.mark.parametrize("with_unit", [True, False])
def test_with_charge(empty_detector: CCD, with_unit: bool):
    """Test assignment with Charge."""
    detector = empty_detector
    data_2d = np.array([[1, 2], [3, 4]], dtype=float)

    if with_unit:
        detector.charge = Quantity(data_2d, unit="electron")
        detector.charge += Quantity(data_2d, unit="electron")
        detector.charge = detector.charge + Quantity(data_2d, unit="electron")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.charge = Quantity(data_2d, unit="photon")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.charge += Quantity(data_2d, unit="photon")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.charge = detector.charge + Quantity(data_2d, unit="photon")

    else:
        detector.charge = data_2d
        detector.charge += data_2d
        detector.charge = detector.charge + data_2d

    np.testing.assert_equal(data_2d, np.array([[1, 2], [3, 4]], dtype=float))
    np.testing.assert_equal(detector.charge, 3 * data_2d)

    result = Quantity(detector.charge)
    np.testing.assert_equal(result.value, 3 * data_2d)
    assert result.unit == "electron"


@pytest.mark.parametrize("with_unit", [True, False])
def test_with_signal(empty_detector: CCD, with_unit: bool):
    """Test assignment with Signal."""
    detector = empty_detector
    data_2d = np.array([[1, 2], [3, 4]], dtype=float)

    if with_unit:
        detector.signal = Quantity(data_2d, unit="V")
        detector.signal += Quantity(data_2d, unit="volt")
        detector.signal = detector.signal + Quantity(1000 * data_2d, unit="mV")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.signal = Quantity(data_2d, unit="adu")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.signal += Quantity(data_2d, unit="adu")

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.signal = detector.signal + Quantity(1000 * data_2d, unit="adu")

    else:
        detector.signal = data_2d
        detector.signal += data_2d
        detector.signal = detector.signal + data_2d

    np.testing.assert_equal(data_2d, np.array([[1, 2], [3, 4]], dtype=float))
    np.testing.assert_equal(detector.signal, 3 * data_2d)

    result = Quantity(detector.signal)
    np.testing.assert_equal(result.value, 3 * data_2d)
    assert result.unit == "V"


@pytest.mark.parametrize("with_unit", [True, False])
def test_with_image(empty_detector: CCD, with_unit: bool):
    """Test assignment with Image."""
    detector = empty_detector
    data_2d = np.array([[1, 2], [3, 4]], dtype=float)

    if with_unit:
        detector.image = Quantity(data_2d, unit="adu").astype(np.uint16)
        detector.image += Quantity(data_2d, unit="adu").astype(np.uint16)
        detector.image = detector.image + Quantity(data_2d, unit="adu").astype(
            np.uint16
        )

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.image = Quantity(data_2d, unit="V").astype(np.uint16)

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.image += Quantity(data_2d, unit="V").astype(np.uint16)

        with pytest.raises(TypeError, match="not compatible with expected unit"):
            detector.image = detector.image + Quantity(data_2d, unit="V").astype(
                np.uint16
            )

    else:
        detector.image = data_2d.astype(np.uint16)
        detector.image += data_2d.astype(np.uint16)
        detector.image = detector.image + data_2d.astype(np.uint16)

    np.testing.assert_equal(data_2d, np.array([[1, 2], [3, 4]], dtype=float))
    np.testing.assert_equal(detector.image, 3 * data_2d.astype(np.uint16))

    result = Quantity(detector.image)
    np.testing.assert_equal(result.value, 3 * data_2d.astype(np.uint16))
    assert result.unit == "adu"
