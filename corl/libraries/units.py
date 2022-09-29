"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Module containing unit dimensions and functions to convert between units
"""
import enum
import math
import typing
from functools import lru_cache

from pydantic import BaseModel, StrictFloat, StrictInt, root_validator, validator


class Distance(enum.Enum):
    """Distance dimension
    """
    Meter = (1.0, ["m", "meter", "meters"])
    Feet = (3.28084, ["ft", "feet"])
    Nautical_Mile = (1 / 1852, ["nm"])
    DEFAULT = Meter


class Angle(enum.Enum):
    """Angle dimension
    """
    Degree = (180.0, ["deg", "degree", "degrees"])
    Rad = (math.pi, ["rad", "radian", "radians"])
    DEFAULT = Rad


class Time(enum.Enum):
    """Time dimension
    """
    Second = (1.0, ["s", "sec", "second", "seconds"])
    Hour = (0.0002777778, ["hr", "hour"])
    DEFAULT = Second


class Speed(enum.Enum):
    """Speed dimension
    """
    Meter_per_Sec = (1, ["m/s", "m_s"])
    Knots = (1.94384, ["knot", "knots", "kts"])
    Feet_per_Min = (196.8504, ["ft/min", "feet/min"])
    Feet_per_Sec = (3.28084, ["ft/s", "feet/second", "feet/sec"])
    DEFAULT = Knots


class MachSpeed(enum.Enum):
    """Mach speed dimension
    """
    Mach = (1.0, ["mach", "Ma"])
    DEFAULT = Mach


class CalibratedSpeed(enum.Enum):
    """Speed dimension for calibrated airspeed
    """
    Meter_per_Sec = (Speed.Meter_per_Sec.value[0], ["mpscas"])
    Knots = (Speed.Knots.value[0], ["kcas"])
    Feet_per_Min = (Speed.Feet_per_Min.value[0], ["fpmcas"])
    Feet_per_Sec = (Speed.Feet_per_Sec.value[0], ["fpscas"])
    DEFAULT = Knots


class IndicatedSpeed(enum.Enum):
    """Speed dimension for indicated airspeed
    """
    Meter_per_Sec = (Speed.Meter_per_Sec.value[0], ["mpsias"])
    Knots = (Speed.Knots.value[0], ["kias"])
    Feet_per_Min = (Speed.Feet_per_Min.value[0], ["fpmias"])
    Feet_per_Sec = (Speed.Feet_per_Sec.value[0], ["fpsias"])
    DEFAULT = Knots


class TrueSpeed(enum.Enum):
    """Speed dimension for true airspeed
    """
    Meter_per_Sec = (Speed.Meter_per_Sec.value[0], ["mpstas"])
    Knots = (Speed.Knots.value[0], ["ktas"])
    Feet_per_Min = (Speed.Feet_per_Min.value[0], ["fpmtas"])
    Feet_per_Sec = (Speed.Feet_per_Sec.value[0], ["fpstas"])
    DEFAULT = Knots


class AngularSpeed(enum.Enum):
    """Angular speed dimension
    """
    radians_per_sec = (1, ["rad/s", "r/s"])
    degrees_per_sec = (57.2958, ["deg/s"])
    DEFAULT = radians_per_sec


class Acceleration(enum.Enum):
    """Acceleration
    """
    knots_per_sec = (1.0, ["knots/s"])
    meter_per_sec_2 = (0.51444563, ["m/s^2"])
    feet_per_sec_2 = (1.68780986, ["ft/s^2"])
    standard_gravity = (0.05245885496, ["g", "G", "gravity", "standard_gravity"])
    DEFAULT = knots_per_sec


class Weight(enum.Enum):
    """Weight
    """
    Kilogram = (1.0, ["kg", "kilogram"])
    Pound = (2.20462, ["lb", "lbs", "pound", "pounds"])
    DEFAULT = Kilogram


class PartOfWhole(enum.Enum):
    """PartOfWhole
    """
    Fraction = (1.0, ["fraction"])
    Percent = (100.0, ["percent"])
    DEFAULT = Fraction


class Force(enum.Enum):
    """Force
    """
    Newton = (1.0, ["N", "newton", "newtons"])
    PoundForce = (0.22481, ["pound-force"])
    DEFAULT = Newton


class NoneUnitType(enum.Enum):
    """None
    """
    NoneUnit = (1.0, ["N/A", "None", "none"])
    DEFAULT = NoneUnit


Dimensions = [
    Distance,
    Angle,
    Time,
    Speed,
    MachSpeed,
    CalibratedSpeed,
    IndicatedSpeed,
    TrueSpeed,
    AngularSpeed,
    Acceleration,
    Weight,
    PartOfWhole,
    Force,
    NoneUnitType
]


@lru_cache(maxsize=50)
def GetUnitFromStr(string: str) -> enum.Enum:
    """Given a string determine the unit that string corresponds to.
    """
    for Dimension in Dimensions:
        for Unit in Dimension:
            if string in Unit.value[1]:
                return Unit
    raise RuntimeError(f"{string} could not be matched to any known unit, please check act3/util/units.py for potential units")


@lru_cache(maxsize=50)
def GetStrFromUnit(unit: enum.Enum) -> str:
    """Given a unit determine the string that unit corresponds to.
    """
    return unit.value[1][0]


def Convert(value: float, from_unit: typing.Union[str, enum.Enum], to_unit: typing.Union[str, enum.Enum]) -> float:
    """Convert a value from a unit to another unit

    Arguments:
        value {float} -- Value to convert
        from_unit {typing.Union[str, enum.Enum]} -- Unit that provided value is in
        to_unit {typing.Union[str, enum.Enum]} -- Desired Unit

    Raises:
        RuntimeError: Thrown if the unit's dimensions do not match

    Returns:
        float -- Converted value
    """

    if isinstance(from_unit, str):
        from_unit = GetUnitFromStr(from_unit)
    if isinstance(to_unit, str):
        to_unit = GetUnitFromStr(to_unit)

    if isinstance(from_unit, type(to_unit)) is False:
        raise RuntimeError(f"Dimensions do not match! {from_unit} -> {to_unit}")

    return value * to_unit.value[0] / from_unit.value[0]  # type: ignore


def ConvertToDefault(value: float, from_unit: typing.Union[None, str, enum.Enum]) -> float:
    """Convert a value from a unit to the default unit for its type

    Arguments:
        value {float} -- Value to convert
        from_unit {typing.Union[str, enum.Enum]} -- Unit that provided value is in

    Returns:
        float -- Converted value
    """

    if from_unit is None:
        return value

    if isinstance(from_unit, str):
        from_unit = GetUnitFromStr(from_unit)

    to_unit = type(from_unit)['DEFAULT']

    return Convert(value=value, from_unit=from_unit, to_unit=to_unit)


class ValueWithUnits(BaseModel):
    """Wrap a value together with its units

    Attributes
    ----------
    value : typing.Any
        The value
    units : typing.Union[None, enum.Enum]
        The units
    """

    value: typing.Union[StrictInt, StrictFloat, bool, str]
    units: typing.Optional[enum.Enum]

    @validator('value', pre=True)
    def value_validator(cls, v):
        """Validate value"""
        # automatically convert numpy types to native types when needed
        # tolist will convert scalar or array to python native type
        # tolist returns a single value (the scalar) when calling it on a single value
        return getattr(v, "tolist", lambda: v)()

    @validator('units', pre=True)
    def convert_string_units(cls, v):
        """Build units out of string"""
        if isinstance(v, str):
            return GetUnitFromStr(v)
        if v is None:
            return NoneUnitType.NoneUnit
        return v

    @root_validator
    def str_no_units(cls, values):
        """Confirm that string values have no units"""
        if isinstance(values.get('value'), str):
            assert values.get('units') is NoneUnitType.NoneUnit, 'String values must have unit None'
        return values

    def convert_to(self, to_unit: typing.Union[None, str, enum.Enum]) -> float:
        """Convert this value to another unit

        Parameters
        ----------
        to_unit : typing.Union[None, str, enum.Enum]
            unit type to convert to

        Returns
        -------
        numbers.Real
            converted value
        """
        return Convert(value=self.value, from_unit=self.units, to_unit=to_unit)  # type: ignore

    class PrincipalValueNormalization(enum.Enum):
        """Enumeration of the type of Principal Value normalization desired.

        Enumeration Values
        ------------------
        Positive
            Normalize to be within the range [0, T], for periodicity T.
        Centered
            Normalize to be within the range [-T/2, T/2], for periodicity T.
        """
        Positive = enum.auto()
        Centered = enum.auto()

    def as_units(self, units: typing.Union[None, str, enum.Enum]) -> typing.Any:
        """View the number in some other units

        Parameters
        ----------
        units : typing.Union[None, str, enum.Enum]
            The desired units

        Returns
        -------
        float
            The value in the desired units
        """
        if units is None and self.units is None:
            return self.value
        if units is None or self.units is None:
            raise RuntimeError(f'Incompatible units involving None: {units} <> {self.units}')
        if units == self.units:
            return self.value
        assert isinstance(self.value, (int, float))
        return Convert(value=self.value, from_unit=self.units, to_unit=units)

    def convert(self, units: typing.Union[None, str, enum.Enum]) -> 'ValueWithUnits':
        """Convert the internal representation to new units

        Parameters
        ----------
        units : typing.Union[None, str, enum.Enum]
            The desired units
        """
        if units is None and self.units is None:
            return self
        if units is None or self.units is None:
            raise RuntimeError('Incompatible units involving None')

        new_units = GetUnitFromStr(units) if isinstance(units, str) else units
        self.value = self.as_units(new_units)
        self.units = new_units

        return self

    def __add__(self, other: 'ValueWithUnits') -> 'ValueWithUnits':
        """Implement addition"""
        raw_value = self.value + other.as_units(self.units)
        return ValueWithUnits(value=raw_value, units=self.units)

    def __sub__(self, other: 'ValueWithUnits') -> 'ValueWithUnits':
        """Implement subtraction"""
        raw_value = self.value - other.as_units(self.units)
        return ValueWithUnits(value=raw_value, units=self.units)

    def normalize_to_principal_value(self, method) -> None:
        """Normalize an angular unit to its principal value.

        Parameters
        ----------
        method : PrincipalValueNormalization
            The type of normalization desired.
        """

        if not isinstance(self.units, Angle):
            raise ValueError('Cannot normalize non-angular value')

        if not isinstance(self.value, (int, float)):
            raise TypeError('Can only normalize numeric values')
        self.value = typing.cast(typing.Union[int, float], self.value)

        if self.units == Angle.Degree:
            period: float = 360
        elif self.units == Angle.Rad:
            period = 2 * math.pi
        else:
            raise ValueError(f'Fix normalize_to_principal_value for {self.units}')

        # Normalize to [-period, period]
        self.value = self.value % period

        # Normalize to [0, period]
        if self.value < 0:
            self.value += period

        if method == self.PrincipalValueNormalization.Positive:
            return
        if method == self.PrincipalValueNormalization.Centered:
            if self.value > period / 2:
                self.value -= period
            return
        raise ValueError(f'Fix normalize_to_principal_value for {method}')

    def __str__(self):
        if self.units is not None:
            return f'{self.value} {self.units.value[1][0]}'
        return str(self.value)

    def __repr__(self):
        return str(self)
