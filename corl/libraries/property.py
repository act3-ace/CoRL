"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Property Module
"""
import abc
import typing
from collections.abc import KeysView
from typing import TypeAlias

import gymnasium.spaces
import numpy as np
import tree
from gymnasium.spaces.utils import flatdim, unflatten
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from ray.rllib.utils.spaces.repeated import Repeated

from corl.libraries.units import Quantity, corl_quantity

NestedDict: TypeAlias = dict[str, "str | NestedDict"]
NestedQuantity: TypeAlias = dict[str, "NestedQuantity | Quantity"]


class Prop(BaseModel, abc.ABC):
    """Represents the space prop outside of RLLIB"""

    name: str | None = None
    description: str | None = None
    model_config = ConfigDict(validate_default=True)

    def contains(self, sample) -> bool:
        """
        Creates RLLIB space
        """
        return self.create_space().contains(self.remove_units(sample))

    def sample(self) -> Quantity | NestedQuantity:
        """
        Creates RLLIB space
        """
        return self.create_quantity(self.create_space().sample())

    @abc.abstractmethod
    def get_units(self) -> str | NestedDict:
        """
        Gets the unit from this Prop
        """

    @abc.abstractmethod
    def create_space(self, seed=None) -> gymnasium.spaces.Space:
        """
        Creates gymnasium space based of property values
        """

    @abc.abstractmethod
    def create_unit_converted_prop(self, unit: str) -> "Prop":
        """
        This function is responsible for taking the current prop object and
        return, it may raise an exception if the prop is not capable of being a
        leaf node
        """

    @abc.abstractmethod
    def zero_mean(self) -> "Prop":
        """
        This function returns a version of this
        prop where all leaf nodes are scaled to be 0 mean
        """

    @abc.abstractmethod
    def scale(self, scale: float) -> "Prop":
        """
        This function returns a version of this
        prop where all leaf nodes are scaled by a scale factor
        """

    @abc.abstractmethod
    def create_quantity(self, value: dict | (float | (int | (list | np.ndarray)))) -> Quantity | NestedQuantity:
        """
        This function taskes in values and will attempt to create either a Quantity or NestedQuantity
        from it, properly applying units along the way
        """

    def remove_units(self, value: Quantity | NestedQuantity) -> dict | (float | (int | (list | np.ndarray))):  # noqa: PLR6301
        """
        This function takes in quanitites and will attempt to remove the units from the quantity
        """
        return tree.map_structure(lambda x: x.m, value)

    @abc.abstractmethod
    def create_zero_sample(self) -> Quantity | NestedQuantity:
        """
        This function will attempt to return 0 for each leaf node for all properties
        in a tree, as this is usually a safe default.  however if 0 is not in the low
        or high for this space it will return the lowest value possible
        Discrete values will always be the lowest, which should be 0
        """

    @abc.abstractmethod
    def create_low_sample(self) -> Quantity | NestedQuantity:
        """
        This function will attempt to return the lowest possible value for each leaf node
        for all properties in a tree.
        """


class BoxProp(Prop):
    """Represents the multi box outside of RLLIB"""

    dtype: np.dtype | None = np.dtype(np.float32)
    low: typing.Sequence[float] | typing.Sequence[typing.Sequence[float]] | np.ndarray
    high: typing.Sequence[float] | typing.Sequence[typing.Sequence[float]] | np.ndarray
    unit: str
    shape: tuple[int] | tuple[int, int] | None = None

    @field_validator("low", "high", mode="before")
    @classmethod
    def low_float_to_list(cls, v):
        return [v] if isinstance(v, int | float) else v

    @model_validator(mode="after")
    def prop_validator(cls, values):
        """Validate"""
        if not all(hasattr(values, k) for k in ("low", "high", "unit")):
            # Something went wrong, return
            return values

        # Broadcast space as needed
        space = gymnasium.spaces.Box(
            low=np.array(values.low).astype(values.dtype),
            high=np.array(values.high).astype(values.dtype),
            dtype=values.dtype,
            shape=values.shape,
        )
        values.low = space.low.tolist()
        values.high = space.high.tolist()
        values.shape = space.shape
        # Validate dimensions
        # 1D case
        assert len(values.low) == len(values.high), "low and high different length"
        if len(values.low) > 0 and isinstance(values.low[0], list):
            # list of lists 2D case
            for i, _ in enumerate(values.low):
                assert len(values.low[i]) == len(values.high[i]), "low and high list elements different length"

        return values

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    def create_space(self, seed=None) -> gymnasium.spaces.Space:
        """
        Creates RLLIB Box space
        """
        return gymnasium.spaces.Box(
            low=np.array(self.low).astype(self.dtype),
            high=np.array(self.high).astype(self.dtype),
            dtype=self.dtype,  # type: ignore
            shape=self.shape,
            seed=seed,
        )

    def min(self, convert: str | None = None) -> Quantity:  # noqa: A003
        """Min are pulled from sensor properties, converted to output units when necessary"""
        tmp = self.create_quantity(list(self.low))
        assert isinstance(tmp, Quantity)
        if convert:
            tmp = tmp.to(convert)
        return tmp

    def max(self, convert: str | None = None) -> Quantity:  # noqa: A003
        """Max are pulled from sensor properties, converted to output units when necessary"""
        tmp = self.create_quantity(list(self.high))
        assert isinstance(tmp, Quantity)
        if convert:
            tmp = tmp.to(convert)
        return tmp

    def create_converted_space(self, convert: str | None = None) -> gymnasium.spaces.Space:
        """
        Creates RLLIB Box space

        #TODO: seed??
        """
        return gymnasium.spaces.Box(
            low=np.array(self.min(convert).m).astype(self.dtype),
            high=np.array(self.max(convert).m).astype(self.dtype),
            dtype=self.dtype,  # type: ignore
            shape=self.shape,
        )

    def get_units(self):
        return self.unit

    def create_unit_converted_prop(self, unit: str):
        return type(self)(
            dtype=self.dtype,
            low=corl_quantity()(np.asarray(self.low), self.unit).to(unit).m.tolist(),
            high=corl_quantity()(np.asarray(self.high), self.unit).to(unit).m.tolist(),
            unit=unit,
            shape=self.shape,
        )

    def zero_mean(self) -> Prop:
        high = np.asarray(self.high)
        low = np.asarray(self.low)
        mean = (high + low) / 2
        return type(self)(low=(low - mean).tolist(), high=(high - mean).tolist(), dtype=self.dtype, unit=self.unit)

    def scale(self, scale: float) -> Prop:
        return type(self)(
            low=np.multiply(self.low, scale).astype(self.dtype).tolist(),
            high=np.multiply(self.high, scale).astype(self.dtype).tolist(),
            shape=self.shape,
            unit=self.unit,
            dtype=self.dtype,
        )

    def create_quantity(self, value: dict | float | int | list | np.ndarray) -> Quantity | NestedQuantity:
        if isinstance(value, float | int):
            new_value = np.array([value], dtype=self.dtype)
        elif isinstance(value, list):
            new_value = np.array(value, dtype=self.dtype)
        elif isinstance(value, np.ndarray):
            new_value = value.astype(self.dtype)
        else:
            raise RuntimeError(
                f"BoxProp tried to create a quantity from {value}, but BoxProp only knows how to make a quantity from "
                f" float, int, list, or np.ndarray, got {type(value)}"
            )
        new_value_shape = new_value.shape
        if self.shape != new_value_shape:
            raise RuntimeError(
                f"BoxProp tried to create a quantity from {value}, but the value given to create_quantity has a shape of {new_value_shape}"
                f" while this BoxProp is expecting to make shapes of {self.shape}"
            )
        return corl_quantity()(new_value, self.get_units())

    def create_zero_sample(self):
        new_value = np.asarray(self.low)
        new_value[new_value < 0.0] = 0.0
        assert np.all(new_value < np.asarray(self.high)), "0 sample ended up higher than high of space, use low_sample instead"
        return self.create_quantity(new_value)

    def create_low_sample(self):
        return self.min()


class DiscreteProp(Prop):
    """Represents the Discrete outside of RLLIB"""

    n: int

    def create_space(self, seed=None) -> gymnasium.spaces.Space:
        """
        Creates RLLIB Discrete space
        """
        return gymnasium.spaces.Discrete(self.n, seed=seed)

    def get_units(self):  # noqa: PLR6301
        return "dimensionless"

    def create_unit_converted_prop(self, unit: str):
        if unit != "dimensionless":
            raise RuntimeError(f"DiscreteProp was told to try and convert to unit {unit} but the only value unit is 'dimensionless'")
        return self

    def zero_mean(self) -> Prop:
        return type(self)(n=self.n)

    def scale(self, scale: float) -> Prop:
        return type(self)(n=self.n)

    def create_quantity(self, value):
        if not (isinstance(value, np.integer | int) or (isinstance(value, np.ndarray) and value.shape == ())):
            raise RuntimeError(f"Discrete Prop create_quantity can only handle integers, got {value=}, {type(value)}")
        return corl_quantity()(value, self.get_units())

    def create_zero_sample(self):
        return self.create_quantity(0)

    def create_low_sample(self):
        return self.create_zero_sample()


class MultiBinary(Prop):
    """Represents the multi binary outside of RLLIB"""

    n: int

    def create_space(self, seed=None) -> gymnasium.spaces.Space:
        """
        Creates RLLIB MultiBinary space
        """
        # TODO ray2 temporarily doesn't support multibinary correctly
        # so just map it to a vector of multi discretes
        return gymnasium.spaces.MultiDiscrete([2] * self.n, seed=seed, dtype=np.int8)

    def get_units(self):  # noqa: PLR6301
        return "dimensionless"

    def create_unit_converted_prop(self, unit: str):
        if unit != "dimensionless":
            raise RuntimeError(f"MultiBinary was told to try and convert to unit {unit} but the only value unit is 'dimensionless'")
        return self

    def zero_mean(self) -> Prop:
        return type(self)(n=self.n)

    def scale(self, scale: float) -> Prop:
        return type(self)(n=self.n)

    def create_quantity(self, value):
        assert isinstance(value, list | np.ndarray)
        return corl_quantity()(value, self.get_units())

    def create_zero_sample(self):
        return self.create_quantity([0] * self.n)

    def create_low_sample(self):
        return self.create_zero_sample()


class CorlRepeated(Repeated):  # noqa: PLW1641
    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return self.child_space.is_np_flattenable

    def __eq__(self, other):
        return isinstance(other, Repeated) and self.child_space == other.child_space and self.max_len == other.max_len


@flatdim.register(CorlRepeated)
def _flatdim_dict(space: CorlRepeated) -> int:
    if space.is_np_flattenable:
        # length of repeated fields is the +1
        return flatdim(space.child_space) * space.max_len + 1
    raise ValueError(f"{space} cannot be flattened to a numpy array, probably because it contains a `Graph` or `Sequence` subspace")


@unflatten.register(CorlRepeated)
def _unflatten_corl_repeated(space: CorlRepeated, x: NDArray[typing.Any]) -> list[dict[str, typing.Any]]:
    # length of repeated fields is the +1
    entries = x[0]
    x = x[1:]
    if space.is_np_flattenable:
        dims = np.asarray([flatdim(space.child_space)] * space.max_len, dtype=np.int_)
        list_flattened = np.split(x, np.cumsum(dims[:-1]))
        return [unflatten(space.child_space, item) for item in list_flattened[0 : int(entries)]]
    raise ValueError(f"{space} cannot be unflattened to a {CorlRepeated}, probably because it contains a `Graph` or `Sequence` subspace")


class RepeatedProp(Prop):
    """Represents the multi binary outside of RLLIB"""

    max_len: int
    child_space: dict[str, Prop]

    def create_space(self, seed=None) -> gymnasium.spaces.Space:
        """
        Creates RLLIB Repeated space
        """
        gymnasium_child_space = gymnasium.spaces.Dict({key: value.create_space(seed=seed) for key, value in self.child_space.items()})
        return CorlRepeated(child_space=gymnasium_child_space, max_len=self.max_len)

    def get_units(self):
        return {key: value.get_units() for key, value in self.child_space.items()}

    def create_unit_converted_prop(self, unit: str):  # noqa: PLR6301
        raise RuntimeError(
            f"RepeatedProp was told to try and convert to unit {unit} but the only but a repeated Prop is not "
            f"capable of having a unit as it is not a leaf node.  this error should never happen and is just a sanity check"
            "to show this code path is not supported"
        )

    def zero_mean(self) -> Prop:
        return type(self)(
            child_space={field_name: field_prop.zero_mean() for field_name, field_prop in self.child_space.items()}, max_len=self.max_len
        )

    def scale(self, scale: float) -> Prop:
        return type(self)(
            child_space={field_name: field_prop.scale(scale) for field_name, field_prop in self.child_space.items()}, max_len=self.max_len
        )

    def create_quantity(self, value):
        assert isinstance(value, list)
        tmp_space = DictProp(spaces=self.child_space)
        new_ret = []
        for inner_row in value:
            assert isinstance(inner_row, dict)
            ret = tmp_space.create_quantity(inner_row)
            new_ret.append(ret)
        return new_ret

    def create_zero_sample(self):  # noqa: PLR6301
        return []

    def create_low_sample(self):  # noqa: PLR6301
        return []


class MultiDiscreteProp(Prop):
    """Represents the MultiDiscrete"""

    nvec: np.ndarray | list
    dtype: np.dtype = np.dtype(np.int32)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_space(self, seed=None) -> gymnasium.spaces.Space:
        """
        Creates MultiDiscrete gymnasium space
        """
        return gymnasium.spaces.MultiDiscrete(self.nvec, dtype=self.dtype, seed=seed)  # type: ignore

    def get_units(self):  # noqa: PLR6301
        return "dimensionless"

    def create_unit_converted_prop(self, unit: str):
        if unit != "dimensionless":
            raise RuntimeError(f"MultiDiscreteProp was told to try and convert to unit {unit} but the only value unit is 'dimensionless'")
        return self

    def zero_mean(self) -> Prop:
        return type(self)(nvec=self.nvec, dtype=self.dtype)

    def scale(self, scale) -> Prop:
        return type(self)(nvec=self.nvec, dtype=self.dtype)

    def create_quantity(self, value):
        assert isinstance(value, list | np.ndarray)
        assert len(value) == len(self.nvec)
        return corl_quantity()(value, self.get_units())

    def create_zero_sample(self):
        return self.create_zero_sample([0] * len(self.nvec))

    def create_low_sample(self):
        return self.create_zero_sample()


class DictProp(Prop, typing.Mapping[str, Prop]):
    """Represents the MultiDiscrete"""

    spaces: dict[str, Prop]

    def create_space(self, seed=None) -> gymnasium.spaces.Dict:
        """
        Creates MultiDiscrete gymnasium space
        """
        return gymnasium.spaces.Dict(spaces={key: sub_prop.create_space(seed=seed) for key, sub_prop in self.items()})

    def get_units(self):
        ret = {}
        for key, prop in self.items():
            ret[key] = prop.get_units()
        return ret

    def create_unit_converted_prop(self, unit: str):  # noqa: PLR6301
        raise RuntimeError(
            f"DictProp was told to try and convert to unit {unit} but the only but a DictProp is not "
            f"capable of having a unit as it is not a leaf node.  this error should never happen and is just a sanity check"
            "to show this code path is not supported"
        )

    def zero_mean(self) -> Prop:
        return type(self)(spaces={field_name: field_prop.zero_mean() for field_name, field_prop in self.items()})

    def scale(self, scale) -> Prop:
        return type(self)(spaces={field_name: field_prop.scale(scale) for field_name, field_prop in self.items()})

    def create_quantity(self, value):
        assert isinstance(value, dict)
        ret = {}
        for prop_name, inner_prop in self.items():
            val = value.get(prop_name)
            if val is None:
                raise RuntimeError(
                    "Dict Prop tried to make quantity, but the value provided was not a valid value for this prop"
                    f"{value=}, space is {self.spaces}"
                )
            ret[prop_name] = inner_prop.create_quantity(val)
        return ret

    def create_zero_sample(self):
        ret = {}
        for prop_name, prop in self.items():
            ret[prop_name] = prop.create_zero_sample()

        return ret

    def create_low_sample(self):
        ret = {}
        for prop_name, prop in self.items():
            ret[prop_name] = prop.create_low_sample()

        return ret

    def __getitem__(self, key: str) -> Prop:
        """Get the space that is associated to `key`."""
        return self.spaces[key]

    def keys(self) -> KeysView:
        """Returns the keys of the Dict."""
        return KeysView(self.spaces)

    def __setitem__(self, key: str, value: Prop):
        """Set the space that is associated to `key`."""
        assert isinstance(value, Prop), f"Trying to set {key} to Dict prop with value that is not a Prop, actual type: {type(value)}"
        self.spaces[key] = value

    def __iter__(self):
        """Iterator through the keys of the subspaces."""
        yield from self.spaces

    def __len__(self) -> int:
        """Gives the number of simpler spaces that make up the `Dict` space."""
        return len(self.spaces)
