# pylint: disable=too-many-lines
"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
ENV Space Util Module

this file is a mypy nightmare due to lots of typing unions and complex types
do not trust mypy, trust the unit tests
"""
import abc
import typing

import gym
from pydantic import BaseModel, PyObject, validator

from corl.libraries.env_space_util import EnvSpaceUtil, convert_gym_space, convert_sample


class SpaceTransformationBase(abc.ABC):
    """
    SpaceTransformationBase provides an anstract class to transform space decfintions
    designed ot be used to change space and the boundary of the agent and policy for
    policies that requires specific space types
    """

    def __init__(self, input_space: gym.Space, config: typing.Dict) -> None:
        self.config = self.get_validator(**config)
        self._input_space: gym.Space = input_space
        self._output_space: gym.Space = self.convert_space(self._input_space)

    @abc.abstractmethod
    def convert_space(self, input_space: gym.Space) -> gym.Space:
        """
        Converts input space to transformed space
        """
        raise NotImplementedError

    @abc.abstractmethod
    def convert_sample(self, sample: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        Converts sample from input space to transformed space
        """
        raise NotImplementedError

    @abc.abstractmethod
    def convert_transformed_sample(self, transformed_sample: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        Converts sample from transformed spae back to transformed space
        """
        raise NotImplementedError

    @property
    def input_space(self, ) -> gym.Space:
        """
        Original input space
        """
        return self._input_space

    @property
    def output_space(self, ) -> gym.Space:
        """
        Transformed space
        """
        return self._output_space

    @property
    def get_validator(self) -> typing.Type[BaseModel]:
        """
        Validator for subclass configuration
        """
        return BaseModel


class SpaceTypeConversionValidator(BaseModel):
    """
    Validator for direct space type conversion
    """
    output_type: PyObject

    @validator("output_type")
    def validate_output_type(cls, val):
        """
        Validate output type is a gym.Space
        """
        if issubclass(val, gym.Space):
            return val

        raise TypeError(f"Invalid output_type {val} in gym.Space")


class SpaceTypeConversion(SpaceTransformationBase):
    """
    SpaceTypeConversion is a SpaceTransformationBase that attempts to convert
    a space into specificed gym.Space types
    """

    def __init__(self, input_space: gym.Space, config: typing.Dict) -> None:
        self.config: SpaceTypeConversionValidator
        super().__init__(input_space, config)

    @property
    def get_validator(self) -> typing.Type[SpaceTypeConversionValidator]:
        """
        SpaceTypeConversionValidator used with SpaceTypeConversion
        """
        return SpaceTypeConversionValidator

    def convert_space(self, input_space: gym.Space) -> gym.Space:
        """
        Converts input space to space defined by output_type
        """
        return convert_gym_space(input_space, self.config.output_type)  # type: ignore

    def convert_sample(self, sample: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        Converts sample to space defined by output_type
        """
        return convert_sample(sample=sample, sample_space=self.input_space, output_space=self.output_space)

    def convert_transformed_sample(self, transformed_sample: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        Converts sample in space defined by output_type to the original input space
        """
        return convert_sample(sample=transformed_sample, sample_space=self.output_space, output_space=self.input_space)
