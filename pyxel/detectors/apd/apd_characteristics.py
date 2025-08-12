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

import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from typing_extensions import ReadOnly, Self, deprecated

from pyxel.util import get_size, get_uninitialized_error

if TYPE_CHECKING:
    from pyxel.detectors import APDGeometry


def detector_gain(capacitance: float, roic_gain: float) -> float:
    """Compute the effective gain of an APD detector.

    The gain is calculated from the ROIC gain and the pixel capacitance.

    Parameters
    ----------
    capacitance : float
        Pixel capacitance in Farads.
    roic_gain : float
        Gain of the readout circuit in V/electron.

    Returns
    -------
    float
        Effective detector gain.
    """
    # Late import to speedup start-up time
    import astropy.constants as const

    return roic_gain * (const.e.value / capacitance)


class ConvertValues(TypedDict):
    """Settings to define values provided by the user."""

    values: ReadOnly[list[tuple[float, float]]]
    # filename: str
    # function: str | Callable[[float], float]


class ConvertFilename(TypedDict):
    """Settings to define filename to read."""

    # values: list[tuple[float, float]]
    filename: ReadOnly[str]
    with_header: bool
    # function: str | Callable[[float], float]


class ConvertFunction(TypedDict):
    """Settings to define a function."""

    # values: list[tuple[float, float]]
    # filename: str
    function: ReadOnly[str | Callable[[float], float]]


def create_func_interpolate(xp: np.ndarray, yp: np.ndarray) -> Callable[[float], float]:
    """Create a 1D linear interpolation function.

    Parameters
    ----------
    xp : ndarray
        Array of x-coordinates (must be monotonic increasing).
    yp : ndarray:
        Array of y-coordinates corresponding to `xp`.

    Returns
    -------
    Callable[[float], float]
        Function that interpolates a single float value.
    """
    import numpy as np

    def _func_interpolate(x: float) -> float:
        return float(
            np.interp(
                x,
                xp=np.asarray(xp, dtype=float),
                fp=np.asarray(yp, dtype=float),
            )
        )

    return _func_interpolate


def to_callable(
    dct: ConvertValues | ConvertFilename | ConvertFunction,
) -> Callable[[float], float]:
    if "values" in dct and ("filename" not in dct and "function" not in dct):
        # Case: ConvertValues -> Create an interpolation function from the data points provided.
        # Late import
        import pandas as pd

        values: list[tuple[float, float]] = dct["values"]

        try:
            df = pd.DataFrame(values)
        except Exception as exc:
            raise ValueError("Failed to convert a list of values") from exc

        if len(df.columns) != 2:
            raise ValueError("Values must be have 2-columns")

        if df.empty:
            raise ValueError("There are no values")

        return create_func_interpolate(
            xp=np.asarray(df.iloc[:, 0], dtype=float),
            yp=np.asarray(df.iloc[:, 1], dtype=float),
        )

    elif "filename" in dct and ("values" not in dct and "function" not in dct):
        # Case: ConvertFilename -> load a table from a file
        from pyxel.inputs import load_table_v2

        filename: str = dct["filename"]
        with_header: bool = dct.get("with_header", False)

        try:
            df = load_table_v2(filename, header=with_header)
        except Exception as exc:
            raise ValueError(f"Failed to convert {filename!r}") from exc

        if len(df.columns) != 2:
            raise ValueError(f"File {filename!r} must have exactly two columns")

        if df.empty:
            raise ValueError(f"File {filename!r} is empty")

        func = create_func_interpolate(
            xp=np.asarray(df.iloc[:, 0], dtype=float),
            yp=np.asarray(df.iloc[:, 1], dtype=float),
        )

        return func

    elif "function" in dct and ("values" not in dct and "filename" not in dct):
        # Case: ConvertFunction -> Return a callable
        function: str | Callable[[float], float] = dct["function"]

        if isinstance(function, str):
            # TODO: Check this, this is security-sensitive
            func = eval(function)

            # TODO: Check that it's a 'Callable[[float], float]'
            return func

        elif callable(function):

            # TODO: Check that it's a 'Callable[[float], float]'
            return function
        else:
            raise ValueError("Invalid function specification")

    else:
        raise ValueError("Invalid conversion dictionary format")


class AvalancheNoGain(TypedDict):
    """Settings for APD without 'avalanche_gain'."""

    # avalanche_gain: float    # unit: electron/electron
    pixel_reset_voltage: ReadOnly[float]  # unit: V
    common_voltage: ReadOnly[float]  # unit: V

    # gain_to_bias: ConvertionSettings | None = None
    bias_to_gain: ConvertValues | ConvertFilename | ConvertFunction


class AvalancheNoPRV(TypedDict):
    """Settings for APD without 'pixel_reset_voltage'."""

    avalanche_gain: ReadOnly[float]  # unit: electron/electron
    # pixel_reset_voltage: float   # unit: V
    common_voltage: ReadOnly[float]  # unit: V

    gain_to_bias: ConvertValues | ConvertFilename | ConvertFunction
    bias_to_gain: ConvertValues | ConvertFilename | ConvertFunction


class AvalancheNoCOMMON(TypedDict):
    """Settings for APD without 'common_voltage'."""

    avalanche_gain: ReadOnly[float]  # unit: electron/electron
    pixel_reset_voltage: ReadOnly[float]  # unit: V
    # common_voltage: float   # unit: V

    gain_to_bias: ConvertValues | ConvertFilename | ConvertFunction
    bias_to_gain: ConvertValues | ConvertFilename | ConvertFunction


# TODO: Add a Classmethod to create an instance of AvalancheSettings
#       from one of the TypedDict
#       Add also all properties to get 'avalanche_gain', ...
class AvalancheSettings:
    """Class to store and compute APD gain/bias settings.

    Parameters
    ----------
    avalanche_gain : float
        Gain of the avalanche multiplication stage.
    pixel_reset_voltage : float
        Pixel reset voltage in V.
    common_voltage : float
        Commonn voltage in V.
    gain_to_bias : callable, optional
        Function to convert from 'avalanche gain' to 'avalanche bias' (in V).
    bias_to_gain : callable, optional
        Function to convert fron 'avalanche bias' (in V) to 'avalanche gain'.
    """

    def __init__(
        self,
        avalanche_gain: float,
        pixel_reset_voltage: float,
        common_voltage: float,
        gain_to_bias: Callable[[float], float] | None = None,
        bias_to_gain: Callable[[float], float] | None = None,
    ):

        self._avalanche_gain: float = avalanche_gain
        self._pixel_reset_voltage: float = pixel_reset_voltage
        self._common_voltage: float = common_voltage
        self._gain_to_bias: Callable[[float], float] | None = gain_to_bias
        self._bias_to_gain: Callable[[float], float] | None = bias_to_gain

    @classmethod
    def build(cls, dct: AvalancheNoGain | AvalancheNoPRV | AvalancheNoCOMMON) -> Self:
        """Build an 'AvalancheSettings' instance."""
        if "pixel_reset_voltage" not in dct or "common_voltage" not in dct:
            # Case: AvalancheNoPRV (No 'pixel_reset_voltage') or AvalancheNoCOMMON (No 'common_voltage')
            avalanche_gain: float = dct["avalanche_gain"]
            gain_to_bias: Callable[[float], float] = to_callable(dct["gain_to_bias"])
            bias_to_gain: Callable[[float], float] = to_callable(dct["bias_to_gain"])

            avalanche_bias: float = gain_to_bias(avalanche_gain)

            if (
                "avalanche_gain" in dct
                and "pixel_reset_voltage" in dct
                and "common_voltage" not in dct
            ):
                # Case: AvalancheNoCOMMON (No 'common' voltage)
                pixel_reset_voltage = dct["pixel_reset_voltage"]

                return cls(
                    avalanche_gain=avalanche_gain,
                    pixel_reset_voltage=pixel_reset_voltage,
                    common_voltage=pixel_reset_voltage - avalanche_bias,
                    gain_to_bias=gain_to_bias,
                    bias_to_gain=bias_to_gain,
                )

            elif (
                "avalanche_gain" in dct
                and "common_voltage" in dct
                and "pixel_reset_voltage" not in dct
            ):
                # Case: AvalancheNoPRV (No 'pixel_reset_voltage')
                common_voltage: float = dct["common_voltage"]

                return cls(
                    avalanche_gain=avalanche_gain,
                    pixel_reset_voltage=common_voltage + avalanche_bias,
                    common_voltage=common_voltage,
                    gain_to_bias=gain_to_bias,
                    bias_to_gain=bias_to_gain,
                )

            else:
                raise NotImplementedError

        # else:
        elif (
            "avalanche_gain" not in dct
            and "pixel_reset_voltage" in dct
            and "common_voltage" in dct
        ):
            # AvalancheNoGain
            pixel_reset_voltage = dct["pixel_reset_voltage"]
            common_voltage = dct["common_voltage"]
            bias_to_gain = to_callable(dct["bias_to_gain"])

            return cls(
                avalanche_gain=pixel_reset_voltage - common_voltage,
                pixel_reset_voltage=pixel_reset_voltage,
                common_voltage=common_voltage,
                gain_to_bias=None,
                bias_to_gain=bias_to_gain,
            )

        else:
            raise NotImplementedError

    @property
    def avalanche_gain(self) -> float:
        """Avalanche gain in e-/e-."""
        return self._avalanche_gain

    @property
    def pixel_reset_voltage(self) -> float:
        """Pixel reset voltage in V."""
        return self._pixel_reset_voltage

    @pixel_reset_voltage.setter
    def pixel_reset_voltage(self, value: float) -> None:
        self._pixel_reset_voltage = value

    @property
    def common_voltage(self) -> float:
        """Common voltage in V."""
        return self._common_voltage

    @common_voltage.setter
    def common_voltage(self, value: float) -> None:
        self._common_voltage = value

    @property
    def avalanche_bias(self) -> float:
        """Compute Avalanche bias voltage in V."""
        return self.pixel_reset_voltage - self.common_voltage


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
        bias_to_node: ConvertValues | ConvertFilename | ConvertFunction,
        avalanche_settings: (
            AvalancheNoGain | AvalancheNoPRV | AvalancheNoCOMMON | None
        ) = None,  # TODO: This parameter should be provided
        #####################
        # Common parameters #
        #####################
        quantum_efficiency: float | None = None,  # unit: NA
        full_well_capacity: float | None = None,  # unit: electron
        adc_bit_resolution: int | None = None,
        adc_voltage_range: tuple[float, float] | None = None,  # unit: V
        #########################
        # DEPRECATED parameters #
        #########################
        avalanche_gain: float | None = None,
        pixel_reset_voltage: float | None = None,
        common_voltage: float | None = None,
    ):
        # Build '_avalanche_settings'
        if avalanche_settings is not None:
            self._avalanche_settings: (
                AvalancheNoGain | AvalancheNoPRV | AvalancheNoCOMMON
            ) = avalanche_settings
        else:
            warnings.warn(
                "Parameters 'avalanche_gain', 'pixel_reset_voltage' and 'common_voltage' are deprecated",
                DeprecationWarning,
                stacklevel=1,
            )

            if (
                avalanche_gain is None
                and pixel_reset_voltage is not None
                and common_voltage is not None
            ):
                self._avalanche_settings = {
                    "pixel_reset_voltage": pixel_reset_voltage,
                    "common_voltage": common_voltage,
                    "bias_to_gain": {"function": "xyz"},  # for Saphira
                }
            elif (
                avalanche_gain is not None
                and pixel_reset_voltage is None
                and common_voltage is not None
            ):
                self._avalanche_settings = {
                    "avalanche_gain": avalanche_gain,
                    "common_voltage": common_voltage,
                    "gain_to_bias": {"function": "xyz"},  # for Saphira
                    "bias_to_gain": {"function": "xyz"},  # for Saphira
                }
            elif (
                avalanche_gain is not None
                and pixel_reset_voltage is not None
                and common_voltage is None
            ):
                self._avalanche_settings = {
                    "avalanche_gain": avalanche_gain,
                    "pixel_reset_voltage": pixel_reset_voltage,
                    "gain_to_bias": {"function": "xyz"},  # for Saphira
                    "bias_to_gain": {"function": "xyz"},  # for Saphira
                }

            else:
                raise ValueError

        # Build object 'AvalancheSettings'
        self._bias_to_node: Callable[[float], float] = to_callable(bias_to_node)
        self._avalanche: AvalancheSettings = AvalancheSettings.build(
            self._avalanche_settings
        )

        # if "avalanche_gain" in self._avalanche_settings:
        #     self._avalanche_gain: float = self._avalanche_settings["avalanche_gain"]
        #
        #     if not (1.0 <= self._avalanche_gain <= 1000.0):
        #         raise ValueError("'apd_gain' must be between 1.0 and 1000.0.")
        #
        #         # if pixel_reset_voltage is not None:
        #         #     if common_voltage is not None:
        #         #         raise ValueError(
        #         #             "Please only specify two inputs out of: avalanche gain, pixel reset"
        #         #             " voltage, common voltage."
        #         #         )
        #
        #     self._avalanche_bias: float = self.gain_to_bias(self._avalanche_gain)
        #     self._pixel_reset_voltage: float = self._avalanche_settings[
        #         "pixel_reset_voltage"
        #     ]
        #     self._common_voltage: float = (
        #         self._avalanche_settings["pixel_reset_voltage"]
        #         - self._avalanche_settings["self.avalanche_bias"]
        #     )

        # elif "common_voltage" != None:
        #     self._avalanche_bias = self.gain_to_bias(avalanche_gain)
        #     self._pixel_reset_voltage = common_voltage + self.avalanche_bias
        #     self._common_voltage = common_voltage
        #
        # else:
        #     raise ValueError(
        #         "Only 'avalanche_gain', missing parameter 'pixel_reset_voltage' "
        #         "or 'common_voltage'."
        #     )
        #
        # elif common_voltage is not None:
        #     if pixel_reset_voltage is None:
        #         raise ValueError(
        #             "Only 'common_voltage', missing parameter 'pixel_reset_voltage' or "
        #             "'avalanche_gain'"
        #         )
        #     self._avalanche_bias = pixel_reset_voltage - common_voltage
        #     self._pixel_reset_voltage = pixel_reset_voltage
        #     self._avalanche_gain = self.bias_to_gain(self.avalanche_bias)
        #     self._common_voltage = common_voltage
        #
        # else:
        #     raise ValueError(
        #         "Not enough input parameters provided to calculate avalanche bias!"
        #     )

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

        # TODO: Is it really needed ? or property 'charge_to_volt_conversion' is enough ?
        self._charge_to_volt_conversion: float = detector_gain(
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
            and self._avalanche_settings == other._avalanche_settings
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
    def avalanche_settings(self) -> AvalancheSettings:
        return self._avalanche

    @deprecated("Use '.avalanche_settings.avalanche_gain")
    @property
    def avalanche_gain(self) -> float:
        """Get APD gain."""
        return self._avalanche.avalanche_gain

    @avalanche_gain.setter
    def avalanche_gain(self, value: float) -> None:
        """Set APD gain."""
        raise NotImplementedError
        # self._avalanche.avalanche_gain = value
        # if np.min(value) < 1.0 or np.max(value) > 1000.0:
        #     raise ValueError("'apd_gain' values must be between 1.0 and 1000.")
        # self._avalanche_gain = value
        # self._avalanche_bias = self.gain_to_bias(value)
        # self._common_voltage = self.pixel_reset_voltage - self.avalanche_bias

    @deprecated("Use '.avalanche_settings.pixel_reset_voltage")
    @property
    def pixel_reset_voltage(self) -> float:
        """Get pixel reset voltage."""
        return self._avalanche.pixel_reset_voltage

    @pixel_reset_voltage.setter
    def pixel_reset_voltage(self, value: float) -> None:
        """Set pixel reset voltage."""
        self._avalanche.pixel_reset_voltage = value
        # self._avalanche_bias = value - self.common_voltage
        # self._avalanche_gain = self.bias_to_gain(self.avalanche_bias)
        # self._pixel_reset_voltage = value

    @deprecated("Use '.avalanche_settings.common_voltage")
    @property
    def common_voltage(self) -> float:
        """Get common voltage."""
        return self._avalanche.common_voltage

    @common_voltage.setter
    def common_voltage(self, value: float) -> None:
        """Set common voltage."""
        self._avalanche.common_voltage = value
        # self._avalanche_bias = self.pixel_reset_voltage - value
        # self._avalanche_gain = self.bias_to_gain(self.avalanche_bias)
        # self._common_voltage = value

    @deprecated("Use '.avalanche_settings.avalanche_bias")
    @property
    def avalanche_bias(self) -> float:
        """Get avalanche bias."""
        return self._avalanche.avalanche_bias

    @property
    def roic_gain(self) -> float:
        """Get roic gain."""
        return self._roic_gain

    @property
    def node_capacitance(self) -> float:
        """Compute node capacitance dynamically from avalanche bias."""
        return self.bias_to_node_capacitance(self.avalanche_settings.avalanche_bias)

    @property
    def charge_to_volt_conversion(self) -> float:
        """Compute charge-to-voltage conversion factor."""
        capacitance = self.bias_to_node_capacitance(
            self.avalanche_settings.avalanche_bias
        )
        return detector_gain(capacitance=capacitance, roic_gain=self.roic_gain)

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
            * self.avalanche_settings.avalanche_gain
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

    def bias_to_node_capacitance(self, bias: float) -> float:
        """Compute node capacitance from bias voltage using user-provided function."""
        if self._bias_to_node is None:
            raise ValueError("'bias_to_node_func' must be provided.")

        return self._bias_to_node(bias) * 1e-15  # fF → F

    # def bias_to_gain(self, bias: float) -> float:
    #     """Compute avalanche gain from bias voltage using user-provided function or inverse."""
    #     if self._bias_to_gain_func is not None:
    #         return self._bias_to_gain_func(bias)
    #
    #     if self._gain_to_bias_func is not None:
    #         inverse_func = _invert_function(
    #             _build_interp_func(self._gain_to_bias_func, "gain", "bias"),
    #             x_min=1.0,
    #             x_max=10.0,
    #         )
    #         return inverse_func(bias)
    #
    #     raise ValueError(
    #         "Either 'bias_to_gain_func' or 'gain_to_bias_func' must be provided."
    #     )
    #
    # def gain_to_bias(self, gain: float) -> float:
    #     """Compute avalanche bias from gain using user-provided function or inverse."""
    #     if self._gain_to_bias_func is not None:
    #         return self._gain_to_bias_func(gain)
    #
    #     if self._bias_to_gain_func is not None:
    #         inverse_func = _invert_function(
    #             _build_interp_func(self._bias_to_gain_func, "bias", "gain"),
    #             x_min=0.5,
    #             x_max=10.0,  # You can adjust depending on expected gain range
    #         )
    #         return inverse_func(gain)
    #
    #     raise ValueError(
    #         "Either 'gain_to_bias_func' or 'bias_to_gain_func' must be provided."
    #     )

    def to_dict(self) -> Mapping:
        """Get the attributes of this instance as a `dict`."""
        dct = {
            "quantum_efficiency": self._quantum_efficiency,
            "full_well_capacity": self._full_well_capacity,
            "adc_voltage_range": self._adc_voltage_range,
            "adc_bit_resolution": self._adc_bit_resolution,
            "roic_gain": self._roic_gain,
            "avalanche_settings": self._avalanche_settings,
            "bias_to_node_func": self._bias_to_node,  # TODO: FIx this
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
