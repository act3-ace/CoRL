# pylint: disable=no-self-argument
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
import collections
import enum
import typing

import gym.spaces
import numpy as np
from pydantic import BaseModel, root_validator, validator
from ray.rllib.utils.spaces.repeated import Repeated

from corl.libraries.units import Convert, GetUnitFromStr


class Prop(BaseModel, abc.ABC):
    """Represents the space prop outside of RLLIB
    """
    name: str
    description: str

    class Config:  # pylint: disable=C0115, R0903
        validate_all = True

    @abc.abstractmethod
    def create_space(self, seed=None) -> gym.spaces.Space:  # pylint: disable=unused-argument
        """
        Creates RLLIB space
        """


class BoxProp(Prop):
    """Represents the multi box outside of RLLIB
    """
    dtype: typing.Optional[np.dtype] = np.dtype(np.float32)
    low: typing.Union[typing.Sequence[float], typing.Sequence[typing.Sequence[float]]]
    high: typing.Union[typing.Sequence[float], typing.Sequence[typing.Sequence[float]]]
    unit: typing.Union[typing.Sequence[str], typing.Sequence[typing.Sequence[str]]]
    shape: typing.Optional[typing.Union[typing.Tuple[int], typing.Tuple[int, int]]]

    @validator('unit')
    def unit_validator(cls, v):
        """Validate unit"""
        for item in v:
            if not isinstance(item, str) and isinstance(item, collections.abc.Sequence):
                # list of lists 2D case
                for value in item:
                    assert isinstance(GetUnitFromStr(value), enum.Enum), f'{value} is not valid unit'
            else:
                # 1D case
                assert isinstance(GetUnitFromStr(item), enum.Enum), f'{value} is not valid unit'
        return v

    @root_validator
    def prop_validator(cls, values):
        """Validate"""
        if not all(k in values for k in ('low', 'high', 'unit')):
            # Something went wrong, return
            return values

        # Broadcast space as needed
        space = gym.spaces.Box(
            low=np.array(values['low']).astype(values['dtype']),
            high=np.array(values['high']).astype(values['dtype']),
            dtype=values['dtype'],
            shape=values['shape']
        )
        values['low'] = space.low.tolist()
        values['high'] = space.high.tolist()

        # Validate dimensions
        # 1D case
        assert len(values['low']) == len(values['high']), "low and high different length"
        assert len(values['low']) == len(values['unit']), "low and unit different length"
        if len(values['low']) > 0:
            if isinstance(values['low'][0], list):
                # list of lists 2D case
                for i, _ in enumerate(values['low']):
                    assert len(values['low'][i]) == len(values['high'][i]), "low and high list elements different length"
                    assert len(values['low'][i]) == len(values['unit'][i]), "low and unit list elements different length"

        return values

    class Config:  # pylint: disable=C0115, R0903
        arbitrary_types_allowed = True
        validate_all = True

    def create_space(self, seed=None) -> gym.spaces.Space:
        """
        Creates RLLIB Box space
        """
        return gym.spaces.Box(
            low=np.array(self.low).astype(self.dtype),
            high=np.array(self.high).astype(self.dtype),
            dtype=self.dtype,  # type: ignore
            shape=self.shape,
            seed=seed
        )

    def min(
        self,
        convert: typing.Union[typing.Sequence[str],
                              typing.Sequence[typing.Sequence[str]],
                              typing.Sequence[enum.Enum],
                              typing.Sequence[typing.Sequence[enum.Enum]]]
    ) -> typing.Union[float, typing.Sequence[float], typing.Sequence[typing.Sequence[float]]]:
        """Min are pulled from sensor properties, converted to output units when necessary
        """
        if isinstance(self.low[0], collections.abc.Sequence):
            # list of lists 2D case
            arr2d = []
            for i, value in enumerate(self.low):
                assert isinstance(value, collections.abc.Sequence)
                unit_array = convert[i]
                assert isinstance(unit_array, collections.abc.Sequence)
                out = []
                for j, val in enumerate(value):
                    tmp = Convert(val, self.unit[i][j], unit_array[j])
                    out.append(tmp)
                arr2d.append(out)
            return arr2d
        # 1D case
        arr = []
        for i, value in enumerate(self.low):
            assert isinstance(value, float)
            in_units = self.unit[i]
            assert isinstance(in_units, (str, enum.Enum))
            out_units = convert[i]
            assert isinstance(out_units, (str, enum.Enum))
            tmp = Convert(value, in_units, out_units)
            arr.append(tmp)
        return arr

    def max(
        self,
        convert: typing.Union[typing.Sequence[str],
                              typing.Sequence[typing.Sequence[str]],
                              typing.Sequence[enum.Enum],
                              typing.Sequence[typing.Sequence[enum.Enum]]]
    ) -> typing.Union[float, typing.Sequence[float], typing.Sequence[typing.Sequence[float]]]:
        """Max are pulled from sensor properties, converted to output units when necessary
        """
        if isinstance(self.high[0], collections.abc.Sequence):
            # list of lists 2D case
            arr2d = []
            for i, value in enumerate(self.high):
                assert isinstance(value, collections.abc.Sequence)
                unit_array = convert[i]
                assert isinstance(unit_array, collections.abc.Sequence)
                out = []
                for j, val in enumerate(value):
                    tmp = Convert(val, self.unit[i][j], unit_array[j])
                    out.append(tmp)
                arr2d.append(out)
            return arr2d
        # 1D case
        arr = []
        for i, value in enumerate(self.high):
            assert isinstance(value, float)
            in_units = self.unit[i]
            assert isinstance(in_units, (str, enum.Enum))
            out_units = convert[i]
            assert isinstance(out_units, (str, enum.Enum))
            tmp = Convert(value, in_units, out_units)
            arr.append(tmp)
        return arr

    def create_converted_space(
        self,
        convert: typing.Union[typing.Sequence[str],
                              typing.Sequence[typing.Sequence[str]],
                              typing.Sequence[enum.Enum],
                              typing.Sequence[typing.Sequence[enum.Enum]]]
    ) -> gym.spaces.Space:
        """
        Creates RLLIB Box space
        """
        return gym.spaces.Box(
            low=np.array(self.min(convert)).astype(self.dtype),
            high=np.array(self.max(convert)).astype(self.dtype),
            dtype=self.dtype,  # type: ignore
            shape=self.shape
        )


class DiscreteProp(Prop):
    """Represents the Discrete outside of RLLIB
    """
    n: int

    def create_space(self, seed=None) -> gym.spaces.Space:
        """
        Creates RLLIB Discrete space
        """
        return gym.spaces.Discrete(self.n, seed=seed)


class MultiBinary(Prop):
    """Represents the multi binary outside of RLLIB
    """
    n: int

    def create_space(self, seed=None) -> gym.spaces.Space:
        """
        Creates RLLIB MultiBinary space
        """
        # TODO ray2 temporarily doesn't support multibinary correctly
        # so just map it to a vector of multi discretes
        return gym.spaces.MultiDiscrete([2] * self.n, seed=seed, dtype=np.int8)


class RepeatedProp(Prop):
    """Represents the multi binary outside of RLLIB
    """
    max_len: int
    child_space: typing.Dict[str, Prop]

    def create_space(self, seed=None) -> gym.spaces.Space:
        """
        Creates RLLIB Repeated space
        """
        gym_child_space = gym.spaces.Dict({key: value.create_space(seed=seed) for key, value in self.child_space.items()})
        return Repeated(child_space=gym_child_space, max_len=self.max_len)


class MultiDiscreteProp(Prop):
    """Represents the MultiDiscrete
    """
    nvec: typing.Union[np.ndarray, list]
    dtype = np.int64

    class Config:
        """Allow arbitrary types for Parameter"""
        arbitrary_types_allowed = True

    def create_space(self, seed=None) -> gym.spaces.Space:
        """
        Creates MultiDiscrete gym space
        """
        return gym.spaces.MultiDiscrete(self.nvec, dtype=self.dtype, seed=seed)
