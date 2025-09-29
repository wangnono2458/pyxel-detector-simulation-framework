#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel Pixel class."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import Any, Self, override

from pyxel.data_structure import ArrayBase, PixelNonVolatile, PixelVolatile
from pyxel.data_structure.array import _is_array_initialized
from pyxel.util import convert_unit, get_size

if TYPE_CHECKING:
    import xarray as xr
    from astropy.units import Quantity

    from pyxel.detectors import Geometry


class Pixel:
    """Pixel class defining and storing information of charge packets within pixel (unit: e⁻).

    Accepted array types: ``np.int32``, ``np.int64``, ``np.uint32``, ``np.uint64``,
    ``np.float16``, ``np.float32``, ``np.float64``.

    Examples
    --------
    Create an empty Pixel container:

    >>> import numpy as np
    >>> from pyxel.detectors import CCD, CCDGeometry, Characteristics, Environment
    >>> detector = CCD(
    ...     geometry=CCDGeometry(row=5, col=5),
    ...     characteristics=Characteristics(),
    ...     environment=Environment(),
    ... )
    >>> detector.pixel
    Pixel<UNINITIALIZED, shape=(5, 5)>
    >>> detector.pixel.ndim
    0
    >>> detector.pixel.shape
    ()

    Perform arithmetic operations:

    >>> detector.pixel.array = detector.pixel.array + np.ones(shape=(5, 5))
    >>> detector.pixel += np.ones(shape=(5, 5))

    Using Astropy Quantity and a units:

    >>> from astropy.units import Quantity
    >>> detector.pixel = Quantity(np.ones(shape=(5, 5)), unit="electron")
    >>> detector.pixel += Quantity(np.ones(shape=(5, 5)), unit="electron")
    >>> detector.pixel = detector.pixel + Quantity(
    ...     np.ones(shape=(5, 5)), unit="electron"
    ... )

    Convert to numpy, Quantity or xarray:

    >>> detector.pixel.array
    array([[2., ...., 2.],
           ...,
           [2., ..., 2.]])
    >>> np.array(detector.pixel)
    array([[2., ...., 2.],
           ...,
           [2., ..., 2.]])
    >>> Quantity(detector.pixel)
    <Quantity [[2., ...., 2.],
              ...,
              [2., ..., 2.]] electron>
    >>> detector.pixel.to_xarray()
    <xarray.DataArray 'pixel' (y: 5, x: 5)>
    array([[2., ...., 2.],
           ...,
           [2., ..., 2.]])
    Coordinates:
      * y        (y) int64 0 1 2 3 4
      * x        (x) int64 0 1 2 3 4
    Attributes:
        units:      electron
        long_name:  Pixel
    """

    TYPE_LIST = (
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    )
    NAME = "Pixel"
    UNIT = "e⁻"

    def __init__(self, geo: "Geometry"):
        self._non_volatile = PixelNonVolatile(geo)
        self._volatile = PixelVolatile(geo)

        self._shape = (geo.row, geo.col)
        self._numbytes = 0

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__

        if self._non_volatile._array is not None and self._volatile._array is not None:
            return f"{cls_name}<shape={self.shape}, dtype={self.dtype}>"

        return f"{cls_name}<UNINITIALIZED, shape={self.shape}>"

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self._non_volatile == other._non_volatile
            and self._volatile == other._volatile
        )

    def __iadd__(self, other) -> Self:
        raise RuntimeError

    def __add__(self, other) -> Self:
        raise RuntimeError

    def __array__(self, dtype: np.dtype | None = None):
        return np.asarray(self.array, dtype=dtype)

    @property
    def shape(self) -> tuple[int, int]:
        """Return array shape."""
        num_cols, num_rows = self._shape
        return num_cols, num_rows

    @property
    def ndim(self) -> int:
        """Return number of dimensions of the array."""
        return len(self._shape)

    @property
    def dtype(self) -> np.dtype:
        """Return array data type."""
        return self.array.dtype

    @property
    def unit(self) -> str:
        """Return the unit for this bucket."""
        return "electron"

    @property
    def volatile(self) -> PixelVolatile:
        return self._volatile

    @property
    def non_volatile(self) -> PixelNonVolatile:
        return self._non_volatile

    def empty_non_volatile(self):
        self._non_volatile.empty()

    def empty_volatile(self):
        self._volatile.empty()

    @property
    def array(self) -> np.ndarray:
        """Two-dimensional numpy array storing the data.

        Only accepts an array with the right type and shape.

        Raises
        ------
        ValueError
            Raised if 'array' is not initialized.
        """
        data_2d: np.ndarray | None = None
        if _is_array_initialized(self._volatile._array):
            data_2d = np.array(self._volatile._array)

        if _is_array_initialized(self._non_volatile._array):
            if data_2d is None:
                data_2d = np.array(self._non_volatile._array)
            else:
                data_2d += np.array(self._non_volatile._array)

        if data_2d is None:
            msg: str = self._get_uninitialized_error_message()
            raise ValueError(msg)

        # Force 'data_2d' to be read-only
        data_2d.setflags(write=False)

        return data_2d

    # @array.setter
    # def array(self, value) -> None:
    #     raise NotImplementedError

    @property
    def numbytes(self) -> int:
        """Recursively calculates object size in bytes using `Pympler` library.

        Returns
        -------
        int
            Size of the object in bytes.
        """
        self._numbytes = get_size(self)
        return self._numbytes

    def to_xarray(self, dtype: np.typing.DTypeLike | None = None) -> "xr.DataArray":
        """Convert into a 2D `DataArray` object with dimensions 'y' and 'x'.

        Parameters
        ----------
        dtype : data-type, optional
            Force a data-type for the array.

        Returns
        -------
        DataArray
            2D DataArray objects.

        Examples
        --------
        >>> detector.photon.to_xarray(dtype=float)
        <xarray.DataArray 'photon' (y: 100, x: 100)>
        array([[15149., 15921., 15446., ..., 15446., 15446., 16634.],
               [15149., 15446., 15446., ..., 15921., 16396., 17821.],
               [14555., 14971., 15446., ..., 16099., 16337., 17168.],
               ...,
               [16394., 16334., 16334., ..., 16562., 16325., 16325.],
               [16334., 15978., 16215., ..., 16444., 16444., 16206.],
               [16097., 15978., 16215., ..., 16681., 16206., 16206.]])
        Coordinates:
          * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
          * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
        Attributes:
            units:      Ph
        """
        import xarray as xr

        num_rows, num_cols = self.shape
        if self._non_volatile._array is None and self._volatile._array is None:
            return xr.DataArray()

        rows = xr.DataArray(
            range(num_rows),
            dims="y",
            attrs={"units": convert_unit("pixel"), "long_name": "Row"},
        )
        cols = xr.DataArray(
            range(num_cols),
            dims="x",
            attrs={"units": convert_unit("pixel"), "long_name": "Column"},
        )

        return xr.DataArray(
            np.array(self.array, dtype=dtype),
            name=self.NAME.lower(),
            dims=["y", "x"],
            coords={"y": rows, "x": cols},
            attrs={"units": convert_unit(self.UNIT), "long_name": self.NAME},
        )

    def plot(self, robust: bool = True) -> None:
        """Plot the array using Matplotlib.

        Parameters
        ----------
        robust : bool, optional
            If True, the colormap is computed with 2nd and 98th percentile
            instead of the extreme values.

        Examples
        --------
        >>> detector.photon.plot()

        .. image:: _static/photon_plot.png
        """
        import matplotlib.pyplot as plt

        arr: "xr.DataArray" = self.to_xarray()

        arr.plot.imshow(robust=robust)
        plt.title(self.NAME)

    def _get_uninitialized_error_message(self) -> str:
        """Get an explicit error message for an uninitialized 'array'.

        This method is used in the property 'array' in the ``Array`` parent class.
        """
        example_model = "simple_collection"
        example_yaml_content = """
- name: simple_collection
  func: pyxel.models.charge_collection.simple_collection
  enabled: true
"""
        cls_name: str = self.__class__.__name__
        obj_name = "pixels"
        group_name = "Charge Collection"

        return (
            f"The '.array' attribute cannot be retrieved because the '{cls_name}'"
            " container is not initialized.\nTo resolve this issue, initialize"
            f" '.array' using a model that generates {obj_name} from the "
            f"'{group_name}' group.\n"
            f"Consider using the '{example_model}' model from"
            f" the '{group_name}' group.\n\n"
            "Example code snippet to add to your YAML configuration file "
            f"to initialize the '{cls_name}' container:\n{example_yaml_content}"
        )

    def to_dict(self) -> dict:
        dct: dict = {}
        if isinstance(self._volatile._array, np.ndarray):
            dct["volatile"] = self._volatile._array.copy()

        if isinstance(self._non_volatile._array, np.ndarray):
            dct["non_volatile"] = self._non_volatile._array.copy()

        return dct

    @classmethod
    def from_dict(cls, geometry: "Geometry", data: Mapping[str, Any]) -> Self:
        obj = cls(geo=geometry)

        if "volatile" in data:
            obj.volatile.array = np.array(data["volatile"])

        if "non_volatile" in data:
            obj.non_volatile.array = np.array(data["non_volatile"])

        return obj
