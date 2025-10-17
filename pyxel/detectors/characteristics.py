#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Detector characteristics model.

This module defines the `Characteristics` class, which represents the
intrinsic physical and electronic parameters of a detector. These parameters
include quantum efficiency, charge-to-voltage conversion, pre-amplification,
full-well capacity, ADC resolution, and voltage range.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from pyxel.detectors import ChargeToVoltSettings
from pyxel.util import get_size, get_uninitialized_error

if TYPE_CHECKING:
    from pyxel.detectors import Geometry


def validate_pre_amplification(
    pre_amplification: int | float | dict[str, int | float] | None,
) -> None:
    """Validate the pre-amplification gain value(s).

    Ensure that the pre-amplification gain(s) are within a valid range.

    Parameters
    ----------
    pre_amplification : int, float, dict or None
        Pre-amplifier gain(s). Can be a single value applied to all pixels
        or a dictionary mapping channel names to individual gains. Unit: V/V

    Raises
    ------
    ValueError
        If one or more gain values are outside the valid range.
    TypeError
        If the provided type is unsupported.
    """
    if pre_amplification is None:
        return

    elif isinstance(pre_amplification, int | float) and not (
        0.0 <= pre_amplification <= 10_000.0
    ):
        raise ValueError("'pre_amplification' must be between 0.0 and 10000.0.")

    elif isinstance(pre_amplification, dict):
        for channel, gain in pre_amplification.items():
            if not (0.0 <= gain <= 10_000.0):
                raise ValueError(
                    f"'pre_amplification' must be between 0.0 and 10000.0. for {channel=}"
                )
    else:
        raise TypeError(
            "'pre_amplification' must be a number, a dictionary of numbers or None"
        )


def to_pre_amplification_map(
    pre_amplification: int | float | dict[str, float] | None,
    geometry: "Geometry",
) -> float | np.ndarray | None:
    """Resolve pre-amplification configuration into a number or a 2D array.

    Convert a per-channel pre-amplification gain definition into a 2D array
    matching the detector geometry.
    This allows subsequent models to operate channel-wise on pre-amplification values.

    Parameters
    ----------
    pre_amplification : int, float, dict or None
        Pre-amplifier gain(s). Can be a single value applied to all pixels
        or a dictionary mapping channel names to individual gains. Unit: V/V

    geometry : Geometry
        Detector geometry definition, including channel layout.

    Returns
    -------
    float or ndarray or None

    Raises
    ------
    ValueError
        If the provided channels do not match those in ``geometry``.
    """
    # Check
    if pre_amplification is None:
        return None

    elif isinstance(pre_amplification, int | float):
        return pre_amplification

    elif isinstance(pre_amplification, dict):
        # Perform channel mismatch check after processing all gains
        if geometry.channels is None:
            raise ValueError("Missing parameter '.channels' in Geometry.")

        defined_channels = set(geometry.channels.readout_position.keys())
        input_channels = set(pre_amplification)
        if defined_channels != input_channels:
            raise ValueError(
                "Mismatch between the defined channels in geometry and provided channel gains."
            )

        gain_map_2d = np.zeros(shape=geometry.shape, dtype=float)

        for channel, gain in pre_amplification.items():
            slice_y, slice_x = geometry.get_channel_coord(channel)
            gain_map_2d[slice_y, slice_x] = gain

        return gain_map_2d

    else:
        raise TypeError(
            "'pre_amplification' must be a number, a dictionary of numbers or None"
        )


class Characteristics:
    """Characteristic attributes of the detector.

    Parameters
    ----------
    quantum_efficiency : float, optional
        Quantum efficiency.
    charge_to_volt_conversion : float, optional
        Sensitivity of charge readout. Unit: V/e-
    pre_amplification : float, optional
        Gain of pre-amplifier. Unit: V/V
    full_well_capacity : float, optional
        Full well capacity. Unit: e-
    adc_bit_resolution : int, optional
        ADC bit resolution.
    adc_voltage_range : tuple of floats, optional
        ADC voltage range. Unit: V
    """

    def __init__(
        self,
        quantum_efficiency: float | None = None,  # unit: NA
        charge_to_volt: ChargeToVoltSettings | None = None,  # unit: volt/electron
        pre_amplification: float | dict[str, float] | None = None,  # unit: V/V
        full_well_capacity: float | None = None,  # unit: electron
        adc_bit_resolution: int | None = None,
        adc_voltage_range: tuple[float, float] | None = None,  # unit: V
    ):
        if quantum_efficiency is not None and not (0.0 <= quantum_efficiency <= 1.0):
            raise ValueError("'quantum_efficiency' must be between 0.0 and 1.0.")

        if charge_to_volt_conversion is not None and not (
            0.0 <= charge_to_volt_conversion <= 100.0
        ):
            raise ValueError(
                "'charge_to_volt_conversion' must be between 0.0 and 100.0."
            )

        validate_pre_amplification(pre_amplification)

        if full_well_capacity is not None and not (0.0 <= full_well_capacity <= 1.0e7):
            raise ValueError("'full_well_capacity' must be between 0 and 1e7.")

        if adc_bit_resolution is not None and not (4 <= adc_bit_resolution <= 64):
            raise ValueError("'adc_bit_resolution' must be between 4 and 64.")

        if adc_voltage_range is not None:
            if not isinstance(adc_voltage_range, Sequence):
                raise TypeError("Voltage range must have length of 2.")

            if len(adc_voltage_range) != 2:
                raise ValueError("Voltage range must have length of 2.")

        self._quantum_efficiency: float | None = quantum_efficiency
        self._charge_to_volt_conversion: float | None = charge_to_volt_conversion

        self._pre_amplification: float | dict[str, float] | None = pre_amplification
        self._pre_amplification_map: float | np.ndarray | None = (
            None  # Geometry is not yet defined at this stage
        )

        self._full_well_capacity: float | None = full_well_capacity

        if adc_voltage_range is None:
            volt_range: tuple[float, float] | None = None
        else:
            # Force 'volt_range' to be a tuple of 2 elements
            start_volt, end_volt = adc_voltage_range
            volt_range = (start_volt, end_volt)

        self._adc_voltage_range = volt_range
        self._adc_bit_resolution = adc_bit_resolution

        # Late binding
        self._geometry: "Geometry" | None = None

        self._numbytes = 0

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self._quantum_efficiency == other._quantum_efficiency
            and self._charge_to_volt_conversion == other._charge_to_volt_conversion
            and self._pre_amplification == other._pre_amplification
            and self._full_well_capacity == other._full_well_capacity
            and self._adc_voltage_range == other._adc_voltage_range
            and self._adc_bit_resolution == other._adc_bit_resolution
        )

    def initialize(self, geometry: "Geometry"):
        self._geometry = geometry
        self._pre_amplification_map = to_pre_amplification_map(
            self._pre_amplification, geometry=self._geometry
        )

    @property
    def quantum_efficiency(self) -> float:
        """Get Quantum efficiency."""
        if self._quantum_efficiency is None:
            raise ValueError(
                get_uninitialized_error(
                    name="quantum_efficiency",
                    parent_name="characteristics",
                )
            )

        return self._quantum_efficiency

    @quantum_efficiency.setter
    def quantum_efficiency(self, value: float) -> None:
        """Set Quantum efficiency."""
        # TODO: Refactor this
        if np.min(value) < 0.0 or np.max(value) > 1.0:
            raise ValueError("'quantum_efficiency' values must be between 0.0 and 1.0.")

        self._quantum_efficiency = value

    @property
    def charge_to_volt_conversion(self) -> float:
        """Get charge to volt conversion parameter."""
        if self._charge_to_volt_conversion is None:
            raise ValueError(
                get_uninitialized_error(
                    name="charge_to_volt_conversion",
                    parent_name="characteristics",
                )
            )

        return self._charge_to_volt_conversion

    @charge_to_volt_conversion.setter
    def charge_to_volt_conversion(self, value: float) -> None:
        """Set charge to volt conversion parameter."""
        if not (0.0 <= value <= 100.0):
            raise ValueError(
                "'charge_to_volt_conversion' must be between 0.0 and 100.0."
            )
        self._charge_to_volt_conversion = value

    @property
    def pre_amplification_map(self) -> float | np.ndarray:
        if self._pre_amplification_map is None:
            raise ValueError(
                get_uninitialized_error(
                    name="channels_gain",
                    parent_name="characteristics",
                )
            )

        return self._pre_amplification_map

    @property
    def pre_amplification(self) -> float | dict[str, float]:
        """Get voltage pre-amplification gain."""
        if self._pre_amplification is None:
            raise ValueError(
                get_uninitialized_error(
                    name="pre_amplification",
                    parent_name="characteristics",
                )
            )

        return self._pre_amplification

    @pre_amplification.setter
    def pre_amplification(self, value: float | dict[str, float]) -> None:
        """Set voltage pre-amplification gain."""
        if self._geometry is None:
            raise RuntimeError(
                "There is no 'Geometry' defined ! Please run method '.initialize(...)'"
            )

        validate_pre_amplification(value)
        pre_amplification_map = to_pre_amplification_map(value, geometry=self._geometry)

        self._pre_amplification = value
        self._pre_amplification_map = pre_amplification_map

    @property
    def adc_bit_resolution(self) -> int:
        """Get bit resolution of the Analog-Digital Converter."""
        if self._adc_bit_resolution is None:
            raise ValueError(
                get_uninitialized_error(
                    name="adc_bit_resolution",
                    parent_name="characteristics",
                )
            )

        return self._adc_bit_resolution

    @adc_bit_resolution.setter
    def adc_bit_resolution(self, value: int) -> None:
        """Set bit resolution of the Analog-Digital Converter."""
        self._adc_bit_resolution = value

    @property
    def adc_voltage_range(self) -> tuple[float, float]:
        """Get voltage range of the Analog-Digital Converter."""
        if self._adc_voltage_range is None:
            raise ValueError(
                get_uninitialized_error(
                    name="adc_voltage_range",
                    parent_name="characteristics",
                )
            )

        return self._adc_voltage_range

    @adc_voltage_range.setter
    def adc_voltage_range(self, value: tuple[float, float]) -> None:
        """Set voltage range of the Analog-Digital Converter."""
        self._adc_voltage_range = value

    @property
    def full_well_capacity(self) -> float:
        """Get Full well capacity."""
        if self._full_well_capacity is None:
            raise ValueError(
                get_uninitialized_error(
                    name="full_well_capacity",
                    parent_name="characteristics",
                )
            )

        return self._full_well_capacity

    @full_well_capacity.setter
    def full_well_capacity(self, value: float) -> None:
        """Set Full well capacity."""
        if not (0 <= value <= 10_000_000):
            raise ValueError("'full_well_capacity' must be between 0 and 1e+7.")

        self._full_well_capacity = value

    @property
    def system_gain(self) -> float | np.ndarray:
        """Get system gain."""
        # Late import
        from astropy.units import Quantity

        # TODO: Is it correct ?
        gain: Quantity = (
            self.quantum_efficiency
            * Quantity(self.pre_amplification_map, unit="V/V")
            * Quantity(2**self.adc_bit_resolution, unit="adu")
        ) / (
            np.max(Quantity(self.adc_voltage_range, unit="V"))
            - np.min(Quantity(self.adc_voltage_range, unit="V"))
        )

        return float(gain.to("adu/V").value)

    @property
    def numbytes(self) -> int:
        """Recursively calculates object size in bytes using Pympler library.

        Returns
        -------
        int
            Size of the object in bytes.
        """
        self._numbytes = get_size(self)
        return self._numbytes

    def to_dict(self) -> Mapping:
        """Get the attributes of this instance as a `dict`."""
        return {
            "quantum_efficiency": self._quantum_efficiency,
            "charge_to_volt_conversion": self._charge_to_volt_conversion,
            "pre_amplification": self._pre_amplification,
            "full_well_capacity": self._full_well_capacity,
            "adc_bit_resolution": self._adc_bit_resolution,
            "adc_voltage_range": self._adc_voltage_range,
        }

    @classmethod
    def from_dict(cls, dct: Mapping):
        """Create a new instance from a `dict`."""
        # Late import to speedup start-up time
        from toolz import dicttoolz

        # TODO: This is a simplistic implementation. Improve this.
        # Extract param 'adc_voltage_range'
        param: Iterable[float] | None = dct.get("adc_voltage_range")
        new_dct: Mapping = dicttoolz.dissoc(dct, "adc_voltage_range")

        if param is None:
            adc_voltage_range: tuple[float, float] | None = None
        else:
            adc_voltage_min, adc_voltage_max = tuple(param)
            adc_voltage_range = adc_voltage_min, adc_voltage_max

        return cls(adc_voltage_range=adc_voltage_range, **new_dct)

    def dump(
        self,
    ) -> dict[str, int | float | str | dict[str, float] | list[float] | None]:
        return {
            "quantum_efficiency": self._quantum_efficiency,
            "charge_to_volt_conversion": self._charge_to_volt_conversion,
            "pre_amplification": self._pre_amplification,
            "full_well_capacity": self._full_well_capacity,
            "adc_bit_resolution": self._adc_bit_resolution,
            "adc_voltage_range": (
                list(self._adc_voltage_range) if self._adc_voltage_range else None
            ),
        }
