"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------

Module containing unit dimensions and functions to convert between units
"""

import inspect
import typing
from collections.abc import Callable
from functools import partial

import numpy as np
from numpy.typing import DTypeLike
from pydantic import AfterValidator, BaseModel, ImportString
from pydantic_core import core_schema

application_registry = None


QuantValues = int | float | np.ndarray | np.floating | np.integer | list


class Unit:
    """
    defines a unit for a given dimension
    """

    def __init__(self, dimension: "Dimension", ratio: "float", known_as: tuple[str, ...]):
        self.dimension = dimension
        self.ratio = ratio
        self.known_as = known_as

    def __str__(self) -> str:
        return self.known_as[0]

    def __repr__(self) -> str:
        return str(self)

    def compatible_units(self) -> list["Unit"]:
        return list(self.dimension.unit_map.values())

    def is_compatible_with(self, other: typing.Union["Unit", str]):
        return self.dimension.is_compatible_with(other)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Unit):
            return self.known_as[0] in __value.known_as
        if isinstance(__value, str):
            return __value in self.known_as
        return False

    def __hash__(self) -> int:
        return hash(str(self))


class Dimension:
    """
    class storing the description of a dimension and all possible
    units for that dimension
    """

    def __init__(self, name: str, registry) -> None:
        self.registry = registry
        self.name = name
        self.base_unit: Unit
        self.unit_map: dict[str, Unit] = {}

    def add_base_unit(self, known_as):
        """
        adds a base unit to this dimension that all unit transformations will
        be relative to
        """
        self.base_unit = Unit(dimension=self, ratio=1.0, known_as=known_as)
        for key in known_as:
            self.unit_map[key] = self.base_unit

        return self.base_unit

    def add_unit(self, ratio: float, known_as: tuple[str, ...]):
        """
        defines a unit to this dimension and saves it's ratio to the base_unit
        """

        tmp = Unit(dimension=self, ratio=ratio, known_as=known_as)
        for key in known_as:
            self.unit_map[key] = tmp

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def is_compatible_with(self, unit: str | Unit):
        """
        checks to see if a requested unit is compatible with this dimension
        """
        return self.registry.get_unit(unit).dimension.base_unit == self.base_unit

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d


class Quantity:
    """
    Quantity that ties together a unit objects and some value
    provides tools for converting this united value to other units

    Quantities will convert any QuantValues to an np.ndarray,
    so lists such as [1,2,3] will be output as np.array([1,2,3])
    when users call .m

    an exception to the above rule is inputs of python int or float,
    quantities will extract those values and return the user int and float again
    """

    def __init__(self, ureg: "Registry", value: QuantValues, units: str | Unit, dtype: DTypeLike = None) -> None:
        self.registry = ureg
        if not isinstance(value, QuantValues):  # type: ignore
            raise RuntimeError(f"value is not QuantValues but {value=} {type(value)}")
        self.value = np.asarray(value, dtype=dtype)
        self.item = np.ndim(value) == 0
        self.unit = self.registry.get_unit(units) if isinstance(units, str) else units

    def to(self, new_unit: str | Unit) -> "Quantity":
        """
        converts this Quantity to another Quantity where the value and unit
        is converted to a requested unit
        """
        new_unit_obj = self.registry.get_unit(new_unit)
        if self.unit == new_unit_obj:
            return self
        assert self.unit.is_compatible_with(new_unit_obj)
        ratio = self.unit.ratio / new_unit_obj.ratio
        new_value = (self.value * ratio).astype(self.dtype)
        return type(self)(self.registry, value=new_value, units=new_unit)

    @property
    def dtype(self) -> np.dtype[typing.Any]:
        """Returns the dtype of value"""
        return self.value.dtype

    @property
    def m(self):
        """
        gets the value of the current quantity
        """
        return self.value.item() if self.item else self.value

    def m_as(self, new_unit: str):
        """
        gets the magnitude of the current Quantity converted to
        another unit
        """
        return self.to(new_unit=new_unit).m

    @property
    def u(self) -> Unit:
        """
        gets the current unit
        """
        return self.unit

    @property
    def units(self) -> Unit:
        """
        gets the current unit
        """
        return self.unit

    def __getitem__(self, index) -> "Quantity":
        """
        allows a user to access into a stored array and attempt to
        make a new Quantity out of the value
        """
        try:
            return type(self)(self.registry, self.m[index], self.unit)
        except TypeError as err:
            raise TypeError(f"Neither Quantity object nor its magnitude ({self.m}) supports indexing") from err

    def is_compatible_with(self, unit: str) -> bool:
        """
        checks that this unit and the requested unit have the same base dimension
        """
        return self.registry.get_unit(unit).dimension.base_unit == self.unit.dimension.base_unit

    def __str__(self) -> str:
        return f"{self.m} {self.u}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Quantity) or self.m != __value.m or self.u != __value.u:
            return False
        return True

    def __hash__(self):
        return hash(str(self))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        if application_registry is None:
            corl_set_ureg(self.registry)
        elif self.registry != corl_get_ureg():
            raise RuntimeError("Unpicked a registry but this application already has a registry that does not match this one")

    @staticmethod
    def _serialize(v) -> str:
        return str(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):  # noqa: PLW3201
        serializer = core_schema.plain_serializer_function_ser_schema(cls._serialize, when_used="json")
        if cls is source:
            # Treat bare usage of ImportString (`schema is None`) as the same as ImportString[Any]
            return core_schema.no_info_plain_validator_function(function=cls.validate, serialization=serializer)
        return core_schema.no_info_before_validator_function(function=cls.validate, schema=handler(source), serialization=serializer)

    @classmethod
    def validate(cls, v):
        if isinstance(v, Quantity):
            return v
        if not isinstance(v, dict):
            # msg = "Quantity validator input must either be a Quantity or Dict"
            # raise TypeError(msg)
            return v

        ext_dtype = v.get("dtype")
        quant_dtype = np.dtype(ext_dtype) if not isinstance(ext_dtype, np.dtype) else ext_dtype

        if not ((v.get("unit") is None) ^ (v.get("units") is None)):
            # raise ValueError("Either 'unit' xor 'units' must be set.")
            return v
        unit = v.get("unit")
        units = v.get("units")
        unit = unit if unit is not None else units
        assert isinstance(unit, str)

        value = v.get("value")
        if not isinstance(value, int | float | list):
            # msg = f"The value passed to Quantity validator was not int | float | list, got {value}"
            # raise TypeError(msg)
            return v

        return corl_quantity()(value, units=unit, dtype=quant_dtype)
        # if field and (output_unit := field.field_info.extra.get("output_unit")):
        #     if not is_compatible(tmp.u, output_unit):
        #         msg = f"Invalid output_unit - '{tmp.units}' is not compatible with '{output_unit}'."
        #         raise ValueError(msg)
        #     tmp = tmp.to(output_unit)


def set_req_unit(unit: str):
    return AfterValidator(lambda v: v.to(unit) if unit else v)


def is_compatible(v1: Quantity | Unit | str, v2: Quantity | Unit | str) -> bool:
    """Convenience function for testing compatibility between quantites/units"""
    q1 = v1 if isinstance(v1, Quantity) else corl_quantity()(0, v1)
    q2 = v2 if isinstance(v2, Quantity) else corl_quantity()(0, v2)

    return q1.is_compatible_with(q2.units)


class UregFuncWrapper:
    """
    UregFuncWrapper is a utility class for wrapping a function, and
    auto wrapping the common code pattern of converting quantities to
    a specific unit, and then doing some processing with the magnitudes
    """

    @staticmethod
    def convert_quantity(output_unit: str, quantity: Quantity) -> QuantValues:
        """
        extract a quantities magnitude in a given unit
        """
        return quantity.m_as(output_unit)

    @staticmethod
    def passthrough(x: typing.Any) -> typing.Any:
        """
        passthrough whatever comes in
        """
        return x

    @staticmethod
    def create_quantity(unit: str, value: QuantValues) -> Quantity:
        """
        generates a quantity out of unit and value, order of args is important here
        because create_quantity is bound as a partial
        """
        return corl_quantity()(value, unit)

    def __init__(self, ureg, inputs: tuple[str | None, ...], outputs: tuple[str | None, ...], func: Callable):
        """
        builds a set of processing partials that will convert all input arguments
        to magnitudes of a specified unit, also creates another set that converts
        raw values to outputs of a specified unit

        also supports passthrough of arguments by providing `None` to specific fields
        """
        self.ureg = ureg
        self.inputs_processors: list[Callable] = []
        self.func = func
        for output_unit in inputs:
            tmp: Callable = partial(self.convert_quantity, output_unit) if output_unit else self.passthrough  # type: ignore
            self.inputs_processors.append(tmp)

        self.input_size = len(inspect.signature(self.func).parameters)
        self.non_default_input_size = len(
            [k for k, v in inspect.signature(self.func).parameters.items() if v.default is inspect.Parameter.empty]
        )
        self.output_processors: list[Callable] = []
        for output_unit in outputs:
            tmp = partial(self.create_quantity, output_unit) if output_unit else self.passthrough  # type: ignore
            self.output_processors.append(tmp)

    def __call__(self, *args, **kwargs) -> typing.Any | tuple[typing.Any, ...]:
        """
        will attempt to auto convert inputs to magnitudes and call the wrapped func
        then convert the outputs back to Quantities
        """
        new_args = args + tuple(kwargs.values())
        assert (
            self.non_default_input_size <= len(new_args) <= self.input_size
        ), f"UregFuncWrapper of {self.func.__name__} is expecting {len(self.inputs_processors)} but got {args=}"
        tmp = self.func(*(preprocess(arg) for preprocess, arg in zip(self.inputs_processors, new_args)))
        # handle python convention for intuitivness, if just 1 processor then just output the value
        if len(self.output_processors) == 1:
            return self.output_processors[0](tmp)
        # multi-output case
        return tuple(postprocess(out) for postprocess, out in zip(self.output_processors, tmp))


class Registry:  # noqa: PLW1641
    def __init__(self) -> None:
        self.dimensions: dict[str, Dimension] = {}
        self.units: dict[str, Unit] = {}

    def add_dimension(self, name: str, base_unit_known_as: tuple[str, ...]):
        """
        adds a dimension to this registry, a dimension must have a base unit that all
        transformations will be done relative to

        example: ureg.add_dimension(name="length", base_unit_known_as=("meter", "m", "meters"))

        name: the name of this dimension
        base_unit_known_as: a tuple of strings describing what the base unit will be known as
        """
        tmp = Dimension(name=name, registry=self)
        self.dimensions[name] = tmp
        new_base_unit = tmp.add_base_unit(base_unit_known_as)
        for key in base_unit_known_as:
            self.units[key] = new_base_unit

    def add_unit(self, ratio: float, relative_to: str | Unit, known_as: tuple[str, ...]):
        """
        Adds the specified unit to the registry

        example: ureg.add_unit(1 / 3.28084, "meter", ("feet", "ft", "foot"))

        ratio: the value to multiple the 'relative_to' unit to get this unit
        relative_to: the unit that registry calculations should be done off of
        known_as: a tuple of strings that describe what this unit is called
        """
        relative_to_unit = self.get_unit(relative_to)
        relative_ratio_to_base = relative_to_unit.ratio * ratio
        tmp = Unit(dimension=relative_to_unit.dimension, ratio=relative_ratio_to_base, known_as=known_as)
        for key in known_as:
            self.units[key] = tmp

    def get_unit(self, unit_name: str | Unit) -> Unit:
        """
        gets the specified unit from the registry
        """
        return self.units[str(unit_name)]

    @property
    def Quantity(main_self):
        """
        provides a constructor for a Quantity that already has the ureg argument
        set
        """
        return partial(Quantity, main_self)

    def wraps(self, output_units: tuple[str | None, ...], input_units: tuple[str | None, ...], func: Callable) -> UregFuncWrapper:
        """
        generates a function wrapper class for a specified function
        This wrapper will
        """
        return UregFuncWrapper(self, inputs=input_units, outputs=output_units, func=func)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Registry):
            return self.dimensions.keys() == __value.dimensions.keys() and self.units.keys() == __value.units.keys()
        return False

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        if application_registry is None:
            corl_set_ureg(self)
        elif self != corl_get_ureg():
            raise RuntimeError("Unpicked a registry but this application already has a registry that does not match this one")


def corl_get_ureg() -> Registry:
    """
    gets the global registry for this process
    """
    assert application_registry is not None
    return application_registry


def corl_set_ureg(ureg: Registry):
    """
    sets the global registry for this process
    """
    global application_registry  # noqa: PLW0603
    application_registry = ureg


def corl_quantity():
    """
    corl_quantity short hands the very common pattern of
    corl_get_ureg().Quantity
    """
    return corl_get_ureg().Quantity


def default_unit_definitions(ureg):
    ureg.add_dimension(name="dimensionless", base_unit_known_as=("dimensionless",))

    ureg.add_dimension(name="length", base_unit_known_as=("meter", "m", "meters"))
    ureg.add_unit(1 / 3.28084, "meter", ("feet", "ft", "foot"))
    ureg.add_unit(1852.0018, "meter", ("nautical_mile", "nmi", "nautical_miles"))

    ureg.add_dimension(name="angle", base_unit_known_as=("radian", "rad", "radians"))
    ureg.add_unit(1 / 57.2958, "radian", ("degree", "deg", "degrees"))

    ureg.add_dimension(name="time", base_unit_known_as=("second", "sec", "seconds"))

    # TODO: can we just derive this??
    ureg.add_dimension(name="velocity", base_unit_known_as=("meter/second", "meter / second", "m/s"))
    ureg.add_unit(1 / 1.94384, "m/s", ("knot", "knots"))
    ureg.add_unit(1 / 3.28084, "m/s", ("foot/second", "ft/s", "foot / second"))

    ureg.add_dimension(name="acceleration", base_unit_known_as=("m/s^2", "meter/second^2", "m/s**2", "meter/second**2"))
    ureg.add_unit(9.80665, "m/s^2", ("standard_gravity", "g0"))

    ureg.add_dimension(name="force", base_unit_known_as=("newton", "newtons", "N"))

    ureg.add_dimension(name="angular_velocity", base_unit_known_as=("radian/second", "radian / second", "rad/sec", "rad/s"))
    ureg.add_unit(1 / 57.2958, "radian/second", ("degree / second", "degrees / second", "deg/s", "deg/sec"))


class UnitRegistryConfiguration(BaseModel):
    """
    Validation for arguments related to modifying the
    unit registry, or adding in custom user defined dimensions,unit
    and unit conversions

    programatic_defines: typing.List[ImportString]  -- A list of python paths
       to a function that processes the ureg and does programmatic processing
       of the registry, such as adding custom dimensions, units, or contexts
       *THESE FUNCTIONS MUST TAKE A UREG AS AN ARGUMENT*
    """

    programatic_defines: list[ImportString] = []

    def create_registry_from_args(self) -> Registry:
        """
        takes the configuration arguments provided to this validator
        and uses them to create a registry to return to the
        caller
        """
        ureg = Registry()
        default_unit_definitions(ureg)
        for x in self.programatic_defines:
            x(ureg)

        return ureg


if __name__ == "__main__":
    ureg = UnitRegistryConfiguration().create_registry_from_args()

    ureg2 = UnitRegistryConfiguration().create_registry_from_args()
    ureg2.add_dimension("test", base_unit_known_as=("testing",))

    # import pickle

    # tmp = pickle.dumps(ureg)
    # print(tmp)
    # new_ureg = pickle.loads(tmp)
    # assert application_registry == new_ureg
    # assert application_registry != ureg2
    # print("passed!")

    # _Quant = ureg.Quantity

    # import timeit

    # import numpy as np

    # iter_number = 1000000

    # def raw_bench():
    #     # corl use case
    #     return np.array([1, 2, 3]) * 3.2

    # print("raw bench")
    # print(timeit.timeit(raw_bench, number=iter_number))
    # print("*******************")

    # def basic_bench2():
    #     # corl use case
    #     return _Quant(np.array([1, 2, 3]), "feet").to("feet")

    # print("basic bench same unit")
    # print(timeit.timeit(basic_bench2, number=iter_number))
    # print("*******************")

    # def basic_bench():
    #     # corl use case
    #     return _Quant(np.array([1, 2, 3]), "feet").to("meter")

    # print("basic bench different unit")
    # print(timeit.timeit(basic_bench, number=iter_number))
    # print("*******************")
