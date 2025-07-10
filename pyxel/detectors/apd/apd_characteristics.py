#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW.

References
----------
[1] Leonardo MW Ltd., Electrical Interface Document for SAPHIRA (ME1000) in a 68-pin LLC
(3313-520-), issue 1A, 2012.

[2] I. Pastrana et al., HgCdTe SAPHIRA arrays: individual pixel measurement of charge gain and
node capacitance utilizing a stable IR LED, in High Energy, Optical, and Infrared Detectors for
Astronomy VIII, 2018, vol. 10709, no. July 2018, 2018, p. 37.

[3] S. B. Goebel et al., Overview of the SAPHIRA detector for adaptive optics applications, in
Journal of Astronomical Telescopes, Instruments, and Systems, 2018, vol. 4, no. 02, p. 1.

[4] G. Finger et al., Sub-electron read noise and millisecond full-frame readout with the near
infrared eAPD array SAPHIRA, in Adaptive Optics Systems V, 2016, vol. 9909, no. July 2016, p.
990912.

[5] I. M. Baker et al., Linear-mode avalanche photodiode arrays in HgCdTe at Leonardo, UK: the
current status, in Image Sensing Technologies: Materials, Devices, Systems, and Applications VI,
2019, vol. 10980, no. May, p. 20.
"""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyxel.util import get_size, get_uninitialized_error

if TYPE_CHECKING:
    from pyxel.detectors import APDGeometry


class APDCharacteristics:
    """Characteristic attributes of the APD detector.

    Parameters
    ----------
    roic_gain
        Gain of the read-out integrated circuit. Unit: V/V
    quantum_efficiency : float, optional
        Quantum efficiency.
    full_well_capacity : float, optional
        Full well capacity. Unit: e-
    adc_bit_resolution : int, optional
        ADC bit resolution.
    adc_voltage_range : tuple of floats, optional
        ADC voltage range. Unit: V
    avalanche_gain : float, optional
        APD gain. Unit: electron/electron
    pixel_reset_voltage : float
        DC voltage going into the detector, not the voltage of a reset pixel. Unit: V
    common_voltage : float
        Common voltage. Unit: V
    """

    def __init__(
        self,
        roic_gain: float,  # unit: V
        quantum_efficiency: float | None = None,  # unit: NA
        full_well_capacity: float | None = None,  # unit: electron
        adc_bit_resolution: int | None = None,
        adc_voltage_range: tuple[float, float] | None = None,  # unit: V
        avalanche_gain: float | None = None,  # unit: electron/electron
        pixel_reset_voltage: float | None = None,  # unit: V
        common_voltage: float | None = None,  # unit: V
        detector_type: str = "saphira",
        gain_to_bias_func: Callable[[float], float] | None = None,
        bias_to_gain_func: Callable[[float], float] | None = None,
        bias_to_node_func: Callable[[float], float] | None = None,
    ):
        self._gain_to_bias_func = (
            self._build_interp_func(gain_to_bias_func, "gain", "bias")
            if gain_to_bias_func is not None
            else None
        )
        self._bias_to_gain_func = (
            self._build_interp_func(bias_to_gain_func, "bias", "gain")
            if bias_to_gain_func is not None
            else None
        )
        self._bias_to_node_func = (
            self._build_interp_func(bias_to_node_func, "bias", "capacitance")
            if bias_to_node_func is not None
            else None
        )
        self._original_avalanche_gain: float | None = avalanche_gain
        self._original_pixel_reset_voltage: float | None = pixel_reset_voltage
        self._original_common_voltage: float | None = common_voltage
        self.detector_type = detector_type.lower()

        if self.detector_type not in ("saphira", "lapd", "new"):
            raise ValueError(
                "Invalid detector type. Choose from: 'saphira', 'lapd', 'new'."
            )

        if avalanche_gain is not None:
            self._avalanche_gain: float = avalanche_gain

            if not (1.0 <= avalanche_gain <= 1000.0):
                raise ValueError("'apd_gain' must be between 1.0 and 1000.0.")

            if pixel_reset_voltage is not None:
                if common_voltage is not None:
                    raise ValueError(
                        "Please only specify two inputs out of: avalanche gain, pixel reset"
                        " voltage, common voltage."
                    )

                self._avalanche_bias: float = self.gain_to_bias(avalanche_gain)
                self._pixel_reset_voltage: float = pixel_reset_voltage
                self._common_voltage: float = pixel_reset_voltage - self.avalanche_bias

            elif common_voltage is not None:
                self._avalanche_bias = self.gain_to_bias(avalanche_gain)
                self._pixel_reset_voltage = common_voltage + self.avalanche_bias
                self._common_voltage = common_voltage

            else:
                raise ValueError(
                    "Only 'avalanche_gain', missing parameter 'pixel_reset_voltage' "
                    "or 'common_voltage'."
                )

        elif common_voltage is not None:
            if pixel_reset_voltage is None:
                raise ValueError(
                    "Only 'common_voltage', missing parameter 'pixel_reset_voltage' or "
                    "'avalanche_gain'"
                )
            self._avalanche_bias = pixel_reset_voltage - common_voltage
            self._pixel_reset_voltage = pixel_reset_voltage
            self._avalanche_gain = self.bias_to_gain(self.avalanche_bias)
            self._common_voltage = common_voltage

        else:
            raise ValueError(
                "Not enough input parameters provided to calculate avalanche bias!"
            )

        if quantum_efficiency and not (0.0 <= quantum_efficiency <= 1.0):
            raise ValueError("'quantum_efficiency' must be between 0.0 and 1.0.")

        if adc_bit_resolution and not (4 <= adc_bit_resolution <= 64):
            raise ValueError("'adc_bit_resolution' must be between 4 and 64.")
        if adc_voltage_range and not len(adc_voltage_range) == 2:
            raise ValueError("Voltage range must have length of 2.")
        if full_well_capacity and not (0.0 <= full_well_capacity <= 1.0e7):
            raise ValueError("'full_well_capacity' must be between 0 and 1e7.")

        self._quantum_efficiency: float | None = quantum_efficiency
        self._full_well_capacity: float | None = full_well_capacity
        self._adc_voltage_range: tuple[float, float] | None = adc_voltage_range
        self._adc_bit_resolution: int | None = adc_bit_resolution
        self._node_capacitance: float = self.bias_to_node_capacitance(
            self.avalanche_bias
        )
        self._roic_gain: float = roic_gain
        self._charge_to_volt_conversion: float = self.detector_gain(
            capacitance=self._node_capacitance,
            roic_gain=self.roic_gain,
        )

        # TODO: This variable is available in class 'Characteristics' and 'APDCharacteristics'
        #       Refactor this
        self._channels_gain: float | np.ndarray | None = None

        # Late binding
        self._geometry: "APDGeometry" | None = None

        self._numbytes = 0

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self._quantum_efficiency == other._quantum_efficiency
            and self._full_well_capacity == other._full_well_capacity
            and self._adc_bit_resolution == other._adc_bit_resolution
            and self._adc_voltage_range == other._adc_voltage_range
            and self._avalanche_gain == other._avalanche_gain
            and self._pixel_reset_voltage == other._pixel_reset_voltage
            and self._common_voltage == other._common_voltage
        )

    # TODO: This method exists in class 'Characteristics' and 'APDCharacteristics'
    #       Refactor this
    def _build_channels_gain(self, value: float | dict[str, float] | None):
        if value is None:
            self._channels_gain = None
            return

        if isinstance(value, float):
            if not (0.0 <= value <= 100.0):
                raise ValueError(
                    "'charge_to_volt_conversion' must be between 0.0 and 100.0."
                )
            self._channels_gain = value
            return

        if isinstance(value, dict):
            if self._geometry is None:
                raise ValueError(
                    "Geometry must be initialized before setting channel gains."
                )

            if self._geometry.channels is None:
                raise ValueError("Missing parameter '.channels' in Geometry.")

            value_2d = np.zeros(shape=self._geometry.shape, dtype=float)

            for channel, gain in value.items():
                if not (0.0 <= gain <= 100.0):
                    raise ValueError(
                        f"Gain for channel {channel} must be between 0.0 and 100.0."
                    )
                slice_y, slice_x = self._geometry.get_channel_coord(channel)
                value_2d[slice_y, slice_x] = gain

            self._channels_gain = value_2d

            # Perform channel mismatch check after processing all gains
            defined_channels = set(self._geometry.channels.readout_position.keys())
            input_channels = set(value.keys())
            if defined_channels != input_channels:
                raise ValueError(
                    "Mismatch between the defined channels in geometry and provided channel gains."
                )
            return

        raise TypeError(
            "Invalid type for 'charge_to_volt_conversion'; expected float or dict."
        )

    # TODO: This method is similar in 'APDCharacteristics and 'Characteristics'
    #       Refactor these methods
    def initialize(self, geometry: "APDGeometry"):
        self._geometry = geometry
        charge_to_volt = self.charge_to_volt_conversion
        if charge_to_volt is not None:
            self._build_channels_gain(value=charge_to_volt)

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
        if np.min(value) < 0.0 or np.max(value) > 1.0:
            raise ValueError("'quantum_efficiency' values must be between 0.0 and 1.0.")

        self._quantum_efficiency = value

    @property
    def avalanche_gain(self) -> float:
        """Get APD gain."""
        return self._avalanche_gain

    @avalanche_gain.setter
    def avalanche_gain(self, value: float) -> None:
        """Set APD gain."""
        if np.min(value) < 1.0 or np.max(value) > 1000.0:
            raise ValueError("'apd_gain' values must be between 1.0 and 1000.")
        self._avalanche_gain = value
        self._avalanche_bias = self.gain_to_bias(value)
        self._common_voltage = self.pixel_reset_voltage - self.avalanche_bias

    @property
    def pixel_reset_voltage(self) -> float:
        """Get pixel reset voltage."""
        return self._pixel_reset_voltage

    @pixel_reset_voltage.setter
    def pixel_reset_voltage(self, value: float) -> None:
        """Set pixel reset voltage."""
        self._avalanche_bias = value - self.common_voltage
        self._avalanche_gain = self.bias_to_gain(self.avalanche_bias)
        self._pixel_reset_voltage = value

    @property
    def common_voltage(self) -> float:
        """Get common voltage."""
        return self._common_voltage

    @common_voltage.setter
    def common_voltage(self, value: float) -> None:
        """Set common voltage."""
        self._avalanche_bias = self.pixel_reset_voltage - value
        self._avalanche_gain = self.bias_to_gain(self.avalanche_bias)
        self._common_voltage = value

    @property
    def avalanche_bias(self) -> float:
        """Get avalanche bias."""
        return self._avalanche_bias

    @property
    def roic_gain(self) -> float:
        """Get roic gainn."""
        return self._roic_gain

    @property
    def node_capacitance(self) -> float:
        """Get node capacitance based on detector type."""
        if self.detector_type == "saphira":
            return self.node_capacitance_saphira
        elif self.detector_type == "lapd":
            return self.node_capacitance_lapd
        elif self.detector_type == "new":
            raise NotImplementedError(
                "node_capacitance for 'new' detector is not implemented."
            )
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

    @property
    def node_capacitance_saphira(self) -> float:
        """Get Saphira node capacitance dynamically from avalanche_bias."""
        return self.bias_to_node_capacitance_saphira(self.avalanche_bias)

    @property
    def node_capacitance_lapd(self) -> float:
        """Get LAPD node capacitance dynamically from avalanche_bias."""
        return self.bias_to_node_capacitance_lapd(self.avalanche_bias)

    @property
    def charge_to_volt_conversion(self) -> float:
        """Get charge to voltage conversion factor based on detector type."""
        if self.detector_type == "saphira":
            return self.charge_to_volt_conversion_saphira
        elif self.detector_type == "lapd":
            return self.charge_to_volt_conversion_lapd
        elif self.detector_type == "new":
            if isinstance(self._charge_to_volt_conversion, (float, dict)):
                return self._charge_to_volt_conversion
            return self.charge_to_volt_conversion_new
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

    @property
    def charge_to_volt_conversion_saphira(self) -> float:
        """Compute Saphira charge-to-voltage conversion factor."""
        return self.detector_gain(
            capacitance=self.node_capacitance_saphira,
            roic_gain=self.roic_gain,
        )

    @property
    def charge_to_volt_conversion_lapd(self) -> float:
        """Compute LAPD charge-to-voltage conversion factor."""
        return self.detector_gain(
            capacitance=self.node_capacitance_lapd,
            roic_gain=self.roic_gain,
        )

    @property
    def charge_to_volt_conversion_new(self) -> float:
        """Compute charge-to-voltage conversion factor for 'new' detectors."""
        return self.detector_gain(
            capacitance=self.bias_to_node_capacitance(self.avalanche_bias),
            roic_gain=self.roic_gain,
        )

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
        if not (4 <= value <= 64):
            raise ValueError("'adc_bit_resolution' must be between 4 and 64.")

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
        if not (0.0 <= value <= 1.0e7):
            raise ValueError("'full_well_capacity' must be between 0 and 1e+7.")

        self._full_well_capacity = value

    @property
    def system_gain(self) -> float:
        """Compute the full system gain based on detector type."""
        return (
            self.quantum_efficiency
            * self.avalanche_gain
            * self.charge_to_volt_conversion
            * 2**self.adc_bit_resolution
        ) / (max(self.adc_voltage_range) - min(self.adc_voltage_range))

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

    @staticmethod
    def _build_interp_func(
        data: str | list[tuple[float, float]] | Callable, xname: str, yname: str
    ) -> Callable[[float], float]:
        if callable(data):
            return data
        elif isinstance(data, str):
            df = pd.read_csv(data)
            if xname not in df.columns or yname not in df.columns:
                raise ValueError(f"CSV must contain '{xname}' and '{yname}' columns")
            x = df[xname].values
            y = df[yname].values
        elif isinstance(data, list | tuple):
            x, y = zip(*data, strict=True)
        else:
            raise TypeError(
                "Function input must be callable, CSV path, or list of (x, y) pairs"
            )
        return lambda val: float(np.interp(val, x, y))

    @staticmethod
    def _invert_function(
        func: Callable[[float], float], x_min: float, x_max: float
    ) -> Callable[[float], float]:
        from scipy.optimize import brentq

        def inverse(y: float) -> float:
            try:
                return float(brentq(lambda x: func(x) - y, x_min, x_max))
            except ValueError:
                raise ValueError(
                    f"Cannot invert function in range [{x_min}, {x_max}] for value {y}"
                ) from None

        return inverse

    def bias_to_node_capacitance(self, bias: float) -> float:
        if self.detector_type == "saphira":
            return self.bias_to_node_capacitance_saphira(bias)
        elif self.detector_type == "lapd":
            return self.bias_to_node_capacitance_lapd(bias)
        elif self.detector_type == "new":
            return self._bias_to_node_func(bias) * 1e-15
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

    @staticmethod
    def bias_to_node_capacitance_saphira(bias: float) -> float:
        """Pixel integrating node capacitance in F.

        The below interpolates empirical published data, however note that
        Node C = Charge Gain / Voltage Gain
        So can be calculated by measuring V gain (varying PRV) and chg gain (PTC); see [2]

        Parameters
        ----------
        bias: float

        Returns
        -------
        output_capacitance: float
        """
        if bias < 1:
            raise ValueError(
                "Warning! Node capacitance calculation is inaccurate for bias voltages"
                " <1 V!"
            )

        # From [2] (Mk13 ME1000; data supplied by Leonardo):
        bias_list = [1, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 10.5]
        capacitance = [46.5, 41.3, 37.3, 34.8, 33.2, 31.4, 30.7, 30.4]

        output_capacitance = float(np.interp(x=bias, xp=bias_list, fp=capacitance))

        return output_capacitance * 1.0e-15

    @staticmethod
    def bias_to_node_capacitance_lapd(bias: float) -> float:
        """Pixel integrating node capacitance in F.

        The provided data set is coming from our result at 2 V of bias (24 fF), applying the same exponential fit as shown from the Saphira.
        Note that
        Node C = Charge Gain / Voltage Gain
        So a new value can be calculated by measuring V gain (varying PRV) and chg gain (PTC); see [2]

        Parameters
        ----------
        bias: float

        Returns
        -------
        output_capacitance: float
        """
        if bias < 1:
            raise ValueError(
                "Warning! Node capacitance calculation is inaccurate for bias voltages"
                " <1 V!"
            )

        # From [2] (Mk13 ME1000; data supplied by Leonardo):
        bias_list = [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10]
        capacitance = [27.2, 25.3, 24, 22.3, 21.2, 20.3, 19.7, 19.1, 18.7, 17.9]

        output_capacitance = float(np.interp(x=bias, xp=bias_list, fp=capacitance))

        return output_capacitance * 1.0e-15

    @property
    def bias_to_node_capacitance_new(self) -> Callable[[float], float]:
        """Build callable to get node capacitance in Farads from bias (using user data)."""
        raw_func = self._build_interp_func(
            self._bias_to_node_func, xname="bias", yname="capacitance"
        )
        return lambda b: raw_func(b) * 1e-15  # Convert fF → F

    def bias_to_gain(self, bias: float) -> float:
        if self.detector_type == "saphira":
            return self.bias_to_gain_saphira(bias)
        elif self.detector_type == "lapd":
            return self.bias_to_gain_lapd(bias)
        elif self.detector_type == "new":
            if self._bias_to_gain_func is None:
                raise ValueError(
                    "bias_to_gain_func must be provided for detector_type='new'"
                )
            return self._bias_to_gain_func(bias)
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

    @staticmethod
    def bias_to_gain_saphira(bias: float) -> float:
        """Calculate gain from bias.

        The formula ignores the soft knee between the linear and unity gain ranges,
        but should be close enough. [2] (Mk13 ME1000)

        Parameters
        ----------
        bias : float

        Returns
        -------
        float
            gain
        """

        gain = 1 + 6.42e-4 * (bias**3.85)

        if gain < 1.0:
            gain = 1.0  # Unity gain is lowest

        return gain

    @property
    def bias_to_gain_new(self) -> Callable[[float], float]:
        if self._bias_to_gain_func is not None:
            return self._build_interp_func(self._bias_to_gain_func, "bias", "gain")

        elif self._gain_to_bias_func is not None:
            gain_to_bias = self._build_interp_func(
                self._gain_to_bias_func, "gain", "bias"
            )
            return self._invert_function(gain_to_bias, x_min=1.0, x_max=1000.0)

        else:
            raise ValueError(
                "Neither 'bias_to_gain_func' nor 'gain_to_bias_func' provided for 'new' detector."
            )

    def gain_to_bias(self, gain: float) -> float:
        if self.detector_type == "saphira":
            return self.gain_to_bias_saphira(gain)
        elif self.detector_type == "lapd":
            return self.gain_to_bias_lapd(gain)
        elif self.detector_type == "new":
            if self._gain_to_bias_func is not None:
                return self._build_interp_func(self._gain_to_bias_func, "gain", "bias")(
                    gain
                )
            elif self._bias_to_gain_func is not None:
                bias_to_gain = self._build_interp_func(
                    self._bias_to_gain_func, "bias", "gain"
                )
                inverse_func = self._invert_function(
                    bias_to_gain, x_min=0.5, x_max=100.0
                )
                return inverse_func(gain)
            else:
                raise ValueError(
                    "Either 'gain_to_bias_func' or 'bias_to_gain_func' must be provided for 'new' detector."
                )
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

    @staticmethod
    def gain_to_bias_saphira(gain: float) -> float:
        """Calculate bias from gain.

        The formula ignores the soft knee between the linear and
        unity gain ranges, but should be close enough. [2] (Mk13 ME1000)

        Parameters
        ----------
        gain: float

        Returns
        -------
        bias: float
        """

        bias = (2.17 * math.log2(gain)) + 2.65

        return bias

    @staticmethod
    def gain_to_bias_lapd(gain: float) -> float:
        """Calculate bias from gain.

        The formula ignores the soft knee between the linear and
        unity gain ranges, but should be close enough. [2] (Mk13 ME1000)

        Parameters
        ----------
        gain: float

        Returns
        -------
        bias: float
        """

        bias = ((gain - 1) / 6.42e-4) ** (1 / 3.85)

        return bias

    @property
    def gain_to_bias_new(self) -> Callable[[float], float]:
        if self._gain_to_bias_func is not None:
            return self._build_interp_func(self._gain_to_bias_func, "gain", "bias")

        elif self._bias_to_gain_func is not None:
            bias_to_gain = self._build_interp_func(
                self._bias_to_gain_func, "bias", "gain"
            )
            return self._invert_function(
                bias_to_gain, x_min=0.5, x_max=10.0
            )  # You can adjust range

        else:
            raise ValueError(
                "Neither 'gain_to_bias_func' nor 'bias_to_gain_func' provided for 'new' detector."
            )

    @staticmethod
    def detector_gain(capacitance: float, roic_gain: float) -> float:
        """Saphira detector gain.

        Parameters
        ----------
        capacitance: float
        roic_gain: float

        Returns
        -------
        float
        """
        # Late import to speedup start-up time
        import astropy.constants as const

        return roic_gain * (const.e.value / capacitance)

    def to_dict(self) -> Mapping:
        """Get the attributes of this instance as a `dict`."""
        dct = {
            "avalanche_gain": self._original_avalanche_gain,
            "common_voltage": self._original_common_voltage,
            "pixel_reset_voltage": self._original_pixel_reset_voltage,
            "quantum_efficiency": self._quantum_efficiency,
            "full_well_capacity": self._full_well_capacity,
            "adc_voltage_range": self._adc_voltage_range,
            "adc_bit_resolution": self._adc_bit_resolution,
            "roic_gain": self._roic_gain,
        }

        return dct

    @classmethod
    def from_dict(cls, dct: Mapping):
        """Create a new instance from a `dict`."""
        # Late import to speedup start-up time
        from toolz import dicttoolz

        new_dct: Mapping = dicttoolz.dissoc(dct, "adc_voltage_range")
        adc_voltage_range = dct["adc_voltage_range"]

        if adc_voltage_range is not None:
            adc_voltage_range = tuple(adc_voltage_range)

        return cls(adc_voltage_range=adc_voltage_range, **new_dct)
