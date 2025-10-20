#  Copyright (c) European Space Agency, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""Charge-to-volt utilities."""

from collections.abc import Mapping

import numpy as np
from typing_extensions import Self

from pyxel.inputs import load_image


class Capacitance:
    """Representation of node capacitance(s).

    The capacitance can be provided either as a scalar value (in fF) or as a path to
    a 2D image file.

    Parameters
    ----------
    capacitance : float or str
        Capacitance in fF or a path to an image file containing capacitance data.
    """

    def __init__(self, capacitance: float | np.ndarray | str):
        capacitance_value: float | np.ndarray
        if isinstance(capacitance, str):
            capacitance_value = load_image(capacitance)
        elif isinstance(capacitance, int | float):
            capacitance_value = capacitance
        else:
            raise TypeError

        self._capacitance: float | np.ndarray | str = capacitance
        self._value: float | np.ndarray = capacitance_value

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__

        return f"{cls_name}(capacitance={self._capacitance!r})"

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and np.allclose(self._value, other._value)

    @property
    def capacitance(self) -> float | np.ndarray:
        """Return capacitance in fF."""
        return self._value


class Factor:
    """Representation of a charge-to-voltage conversion factor(s).

    This factor can be provided either as a scalar value (in V / electron) or as a path to
    a 2D image file.

    Parameters
    ----------
    value : float or str
        Conversion factor in V/electron or a path to an image file containing factor values.
    """

    def __init__(self, value: float | np.ndarray | str):
        factor_value: float | np.ndarray
        if isinstance(value, str):
            factor_value = load_image(value)
        elif isinstance(value, int | float):
            if not (0.0 <= value <= 100.0):
                raise ValueError("'charge_to_volt' must be between 0.0 and 100.0.")

            factor_value = value
        else:
            raise TypeError

        self._factor: float | np.ndarray | str = value
        self._value: float | np.ndarray = factor_value

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}(factor={self._factor!r})"

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and np.allclose(self._value, other._value)

    @property
    def factor(self) -> float | np.ndarray:
        """Return charge-to-voltage conversion factor in V/electron."""
        return self._value


class ChargeToVoltSettings:
    """Settings for charge-to-voltage conversion.

    This class can represent either:
        - a **capacitance-based conversion** where the voltage per electron is derived
          from the capacitance.
        - a **direct charge-to-volt factor** provided as a pre-computed V/electron value
    """

    def __init__(
        self,
        value: float | np.ndarray | str | None = None,
        from_capacitance: float | np.ndarray | str | None = None,
    ):
        param: Capacitance | Factor
        if value is not None:
            param = Factor(value)

        elif from_capacitance is not None:
            param = Capacitance(from_capacitance)

        else:
            raise ValueError

        self._param: Capacitance | Factor = param

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}({self._param!r})"

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self._param == other._param

    @property
    def capacitance(self) -> float | np.ndarray:
        """Return capacitance in fF."""
        if isinstance(self._param, Factor):
            raise TypeError

        return self._param.capacitance

    @capacitance.setter
    def capacitance(self, value: float | np.ndarray) -> None:
        self._param = Capacitance(value)

    def has_charge_to_volt(self) -> bool:
        """Check if the Charge-to-Volt factor is directly defined and not derived from the capacitance."""
        return isinstance(self._param, Factor)

    @property
    def value(self) -> float | np.ndarray:
        """Return charge-to-voltage conversion factor in V/electron."""
        if isinstance(self._param, Factor):
            return self._param.factor
        else:
            # Late import
            import astropy.constants as consts
            from astropy.units import Quantity

            # Get capacitance in fF
            capacitance = Quantity(self._param.capacitance, "fF")

            # Electric charge of one electron in C / electron
            one_electron = Quantity(consts.e) / Quantity(1, "electron")

            # Convert 'capacitance' into a 'charge_to_volt'
            charge_to_volt: Quantity = (one_electron / capacitance).to("V / electron")

            value: np.ndarray = np.array(charge_to_volt)

            if value.ndim == 0:
                return float(value)
            else:
                return value

    @value.setter
    def value(self, value: float | np.ndarray) -> None:
        self._param = Factor(value)

    def to_dict(self) -> dict:
        if isinstance(self._param, Capacitance):
            return {"from_capacitance": self._param._capacitance}
        else:
            return {"value": self._param._factor}

    @classmethod
    def from_dict(cls, dct: Mapping) -> Self:
        if "from_capacitance" in dct:
            return cls(from_capacitance=dct["from_capacitance"])
        elif "value" in dct:
            return cls(value=dct["value"])
        else:
            raise KeyError
