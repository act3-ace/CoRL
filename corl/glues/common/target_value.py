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
import numpy as np
from pydantic import BaseModel

from corl.glues.base_glue import BaseAgentPlatformGlue, BaseAgentPlatformGlueValidator


class LimitConfigValidator(BaseModel):
    """
    minimum: defines the min the obs of the glue may produce
    maximum: defines the max the obs of the glue may produce
    unit: defines the unit for the limits
    clip: defines if the glue should clip the obs values to between min/max
    """
    minimum: float
    maximum: float
    unit: typing.Optional[str]
    clip: bool = False


class TargetValueValidator(BaseAgentPlatformGlueValidator):
    """
    target_value: the Value this glue should produce this episode
    unit: the unit of the value produced
    """
    target_value: float
    unit: str
    limit: LimitConfigValidator


class TargetValue(BaseAgentPlatformGlue):
    """
    Glue class for an observation of some constant value
    """

    # pylint: disable=too-few-public-methods
    class Fields:
        """
        field data
        """
        TARGET_VALUE = "target_value"

    def __init__(self, **kwargs) -> None:
        self.config: TargetValueValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[TargetValueValidator]:
        return TargetValueValidator

    @lru_cache(maxsize=1)
    def get_unique_name(self) -> str:
        """Provies a unique name of the glue to differentiate it from other glues.
        """
        return f"{self.config.name}TargetValue"

    def invalid_value(self) -> OrderedDict:
        """When invalid return a value of 0

        TODO: this may need to be self.min in the case that the minimum is larger than 0 (i.e. a harddeck)

        Returns:
            OrderedDict -- Dictionary with <FIELD> entry containing 1D array
        """
        d = OrderedDict()
        d[self.Fields.TARGET_VALUE] = np.asarray([self.config.target_value], dtype=np.float32)
        return d

    @lru_cache(maxsize=1)
    def observation_units(self):
        """Return a space dictionary that indicates the units of the observation
        """
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.TARGET_VALUE] = [self.config.unit]
        return d

    @lru_cache(maxsize=1)
    def observation_space(self):
        """Describes observation space for a single scalar continuous value

        Returns:
            gym.spaces.Space -- The rllib policy observation space
        """

        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.TARGET_VALUE
                 ] = gym.spaces.Box(self.config.limit.minimum, self.config.limit.maximum, shape=(1, ), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> OrderedDict:
        """Constructs a portion of the observation space

        Arguments:
            platforms {List[BasePlatform]} -- List of current platforms.
                Gauranteed to be length of 1 and set to self.obs_agent

        Returns:
            OrderedDict -- Dictionary containing observation in <FIELD> key
        """

        d = OrderedDict()
        d[self.Fields.TARGET_VALUE] = np.asarray([self.config.target_value], dtype=np.float32)
        return d

    def action_space(self):
        ...

    def apply_action(self, action, observation, action_space, obs_space, obs_units):
        ...
