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
from typing import Annotated

import gymnasium
from pydantic import AfterValidator, BaseModel, ImportString

from corl.libraries.env_space_util import EnvSpaceUtil, convert_gymnasium_space, convert_sample


class SpaceTransformationBase(abc.ABC):
    """
    SpaceTransformationBase provides an anstract class to transform space decfintions
    designed to be used to change space and the boundary of the agent and policy for
    policies that requires specific space types
    """

    def __init__(self, input_space: gymnasium.Space, config: dict) -> None:
        self.config = self.get_validator()(**config)
        self._input_space: gymnasium.Space = input_space
        self._output_space: gymnasium.Space = self.convert_space(self._input_space)

    @abc.abstractmethod
    def convert_space(self, input_space: gymnasium.Space) -> gymnasium.Space:
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
    def input_space(self) -> gymnasium.Space:
        """
        Original input space
        """
        return self._input_space

    @property
    def output_space(self) -> gymnasium.Space:
        """
        Transformed space
        """
        return self._output_space

    @staticmethod
    def get_validator() -> type[BaseModel]:
        """
        Validator for subclass configuration
        """
        return BaseModel


def verify_gym_space(val):
    assert issubclass(val, gymnasium.Space), f"Invalid output_type {val} in gymnasium.Space"
    return val


class SpaceTypeConversionValidator(BaseModel):
    """
    Validator for direct space type conversion
    """

    output_type: Annotated[ImportString, AfterValidator(verify_gym_space)]


class SpaceTypeConversion(SpaceTransformationBase):
    """
    SpaceTypeConversion is a SpaceTransformationBase that attempts to convert
    a space into specified gymnasium.Space types
    """

    def __init__(self, input_space: gymnasium.Space, config: dict) -> None:
        self.config: SpaceTypeConversionValidator
        super().__init__(input_space, config)

    @staticmethod
    def get_validator() -> type[SpaceTypeConversionValidator]:
        """
        SpaceTypeConversionValidator used with SpaceTypeConversion
        """
        return SpaceTypeConversionValidator

    def convert_space(self, input_space: gymnasium.Space) -> gymnasium.Space:
        """
        Converts input space to space defined by output_type
        """
        return convert_gymnasium_space(input_space, self.config.output_type)

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
