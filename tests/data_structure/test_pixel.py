#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import numpy as np
import pytest
from astropy.units import Quantity

from pyxel.data_structure import Pixel, PixelNonVolatile, PixelVolatile
from pyxel.detectors import Geometry


def test_empty_pixel():
    """Test an empty 'pixel' container."""
    # Create an empty pixel
    empty_pixel = Pixel(geo=Geometry(row=2, col=3))
    another_empty_pixel = Pixel(geo=Geometry(row=2, col=3))

    assert repr(empty_pixel) == "Pixel<UNINITIALIZED, shape=(2, 3)>"
    assert empty_pixel == another_empty_pixel

    assert empty_pixel.shape == (2, 3)
    assert empty_pixel.ndim == 2
    assert empty_pixel.unit == "electron"
    assert empty_pixel.numbytes > 0

    with pytest.raises(
        ValueError,
        match=r"\'PixelNonVolatile\' and \'PixelVolatile\' containers are not initialized",
    ):
        _ = empty_pixel.array

    with pytest.raises(
        ValueError,
        match=r"\'PixelNonVolatile\' and \'PixelVolatile\' containers are not initialized",
    ):
        _ = empty_pixel.dtype

    volatile = empty_pixel.volatile
    assert isinstance(volatile, PixelVolatile)
    assert repr(volatile) == "PixelVolatile<UNINITIALIZED, shape=(2, 3)>"
    with pytest.raises(
        ValueError, match=r"\'PixelVolatile\' container is not initialized"
    ):
        _ = empty_pixel.volatile.array

    non_volatile = empty_pixel.non_volatile
    assert isinstance(non_volatile, PixelNonVolatile)
    assert repr(non_volatile) == "PixelNonVolatile<UNINITIALIZED, shape=(2, 3)>"
    with pytest.raises(
        ValueError, match=r"\'PixelNonVolatile\' container is not initialized"
    ):
        _ = empty_pixel.non_volatile.array

    # 'array' setter
    with pytest.raises(
        AttributeError,
        match=r"You must set data to \'PixelNonVolatile\' or \'PixelVolatile\' containers",
    ):
        empty_pixel.array = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    # __add__
    with pytest.raises(ValueError, match=r"Bucket \'Pixel\' is read-only"):
        _ = empty_pixel + np.array([[1, 2], [3, 4]], dtype=np.uint16)

    # __iadd__
    with pytest.raises(ValueError, match=r"Bucket \'Pixel\' is read-only"):
        empty_pixel += np.array([[1, 2], [3, 4]], dtype=np.uint16)


def test_pixel():
    """Test pixel."""
    pixel = Pixel(geo=Geometry(row=2, col=3))

    pixel.non_volatile += np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    np.testing.assert_allclose(
        pixel.non_volatile.array, np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    )
    np.testing.assert_allclose(
        pixel.array, np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    )

    pixel.non_volatile += Quantity([[1, 1, 1], [1, 1, 1]], unit="electron")
    np.testing.assert_allclose(
        pixel.non_volatile.array, np.array([[2, 3, 4], [5, 6, 7]], dtype=float)
    )
    np.testing.assert_allclose(
        pixel.array, np.array([[2, 3, 4], [5, 6, 7]], dtype=float)
    )

    pixel.volatile += np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    np.testing.assert_allclose(
        pixel.volatile.array, np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    )
    np.testing.assert_allclose(
        pixel.array, np.array([[3, 5, 7], [9, 11, 13]], dtype=float)
    )

    pixel.volatile += Quantity([[1, 1, 1], [1, 1, 1]], unit="electron")
    np.testing.assert_allclose(
        pixel.volatile.array, np.array([[2, 3, 4], [5, 6, 7]], dtype=float)
    )
    np.testing.assert_allclose(
        pixel.array, np.array([[4, 6, 8], [10, 12, 14]], dtype=float)
    )

    assert repr(pixel) == "Pixel<shape=(2, 3), dtype=float64>"
    pixel.plot()
