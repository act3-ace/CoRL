"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import typing
from collections import OrderedDict
from functools import lru_cache

import gym

from corl.glues.base_glue import BaseAgentPlatformGlue, BaseAgentPlatformGlueValidator
from corl.libraries.units import NoneUnitType
from corl.simulators.base_parts import BasePlatformPart
from corl.simulators.common_platform_utils import get_part_by_name


class ObservePartValidityValidator(BaseAgentPlatformGlueValidator):
    """
    part: which part to find on the platform
    """
    part: str


class ObservePartValidity(BaseAgentPlatformGlue):
    """Glue to observe a part's validity flag
    """

    # pylint: disable=too-few-public-methods
    class Fields:
        """
        Fields in this glue
        """
        VALIDITY_OBSERVATION = "validity_observation"

    @property
    def get_validator(self) -> typing.Type[ObservePartValidityValidator]:
        return ObservePartValidityValidator

    def __init__(self, **kwargs) -> None:
        self.config: ObservePartValidityValidator
        super().__init__(**kwargs)

        self._part: BasePlatformPart = get_part_by_name(self._platform, self.config.part)
        self._part_name: str = self.config.part

    @lru_cache(maxsize=1)
    def get_unique_name(self) -> str:
        """Class method that retrieves the unique name for the glue instance
        """
        return "ObserveValidity_" + self._part_name

    @lru_cache(maxsize=1)
    def observation_units(self):
        """Units of the sensors in this glue
        """
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.VALIDITY_OBSERVATION] = NoneUnitType
        return d

    @lru_cache(maxsize=1)
    def observation_space(self) -> gym.spaces.Space:
        """Observation Space
        """
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.VALIDITY_OBSERVATION] = gym.spaces.Discrete(2)
        return d

    def get_observation(self) -> OrderedDict:
        """Observation Values
        """
        d = OrderedDict()
        d[self.Fields.VALIDITY_OBSERVATION] = int(self._part.valid)

        return d
