#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Pyxel Image class."""

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from pyxel.data_structure import ArrayBase

if TYPE_CHECKING:
    from pyxel.detectors import Geometry


class Image(ArrayBase):
    """Image class defining and storing information of detector image (unit: adu).

    Accepted array types: ``np.uint16``, ``np.uint32``, ``np.uint64``

    Examples
    --------
    Create an empty Image container:

    >>> import numpy as np
    >>> from pyxel.detectors import CCD, CCDGeometry, Characteristics, Environment
    >>> detector = CCD(
    ...     geometry=CCDGeometry(row=5, col=5),
    ...     characteristics=Characteristics(),
    ...     environment=Environment(),
    ... )
    >>> detector.image
    image<UNINITIALIZED, shape=(5, 5)>
    >>> detector.image.ndim
    0
    >>> detector.image.shape
    ()

    Perform arithmetic operations:

    >>> detector.image.array = detector.image.array + np.ones(
    ...     shape=(5, 5), dtype=np.uint16
    ... )
    >>> detector.image += np.ones(shape=(5, 5), dtype=np.uint16)

    Using Astropy Quantity and a units:

    >>> from astropy.units import Quantity
    >>> data_with_unit = Quantity(np.ones(shape=(5, 5)), unit="adu").astype(np.uint16)
    >>> detector.image = data_with_unit
    >>> detector.image += data_with_unit
    >>> detector.image = detector.image + data_with_unit

    Convert to numpy, Quantity or xarray:

    >>> detector.image.array
    array([[2, ...., 2],
           ...,
           [2, ..., 2]])
    >>> np.array(detector.image)
    array([[2, ...., 2],
           ...,
           [2, ..., 2]])
    >>> Quantity(detector.image)
    <Quantity [[2, ...., 2],
              ...,
              [2, ..., 2]] volt>
    >>> detector.image.to_xarray()
    <xarray.DataArray 'image' (y: 5, x: 5)>
    array([[2, ...., 2],
           ...,
           [2, ..., 2]])
    Coordinates:
      * y        (y) int64 0 1 2 3 4
      * x        (x) int64 0 1 2 3 4
    Attributes:
        units:      adu
        long_name:  image
    """

    TYPE_LIST = (
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.uint64),
    )
    NAME = "Image"
    UNIT = "adu"

    def __init__(self, geo: "Geometry"):
        super().__init__(shape=(geo.row, geo.col))

    @override
    def _get_uninitialized_error_message(self) -> str:
        """Get an explicit error message for an uninitialized 'array'.

        This method is used in the property 'array' in the ``Array`` parent class.
        """
        example_model = "simple_amplifier"
        example_yaml_content = """
- name: simple_amplifier
  func: pyxel.models.readout_electronics.simple_amplifier
  enabled: true
"""
        cls_name: str = self.__class__.__name__
        obj_name = "images"
        group_name = "Readout Electronics"

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
