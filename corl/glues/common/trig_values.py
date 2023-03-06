"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import math
import typing
from collections import OrderedDict
from functools import lru_cache

import gym
import numpy as np

from corl.glues.base_wrapper import BaseWrapperGlue, BaseWrapperGlueValidator
from corl.glues.common.observe_sensor import ObserveSensor
from corl.libraries.units import Convert


class TrigValuesValidator(BaseWrapperGlueValidator):
    """
    Validator for TrigValues glue
    """
    wrapped: ObserveSensor
    observation: str = "direct_observation"
    index: int = 0


class TrigValues(BaseWrapperGlue):
    """
    Glue class for an observation of sin & cos of an angle
    """

    class Fields:
        """
        field data
        """
        COS = "cos"
        SIN = "sin"

    def __init__(self, **kwargs):
        self.config: TrigValuesValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[TrigValuesValidator]:
        return TrigValuesValidator

    @lru_cache(maxsize=1)
    def get_unique_name(self) -> str:
        """Provies a unique name of the glue to differentiate  it from other glues.
        """
        return f"{self.config.name}"

    @lru_cache(maxsize=1)
    def observation_units(self):
        """Return a space dictionary that indicates the units of the observation
        """
        d = gym.spaces.dict.Dict()
        d[self.Fields.COS] = ["None"]
        d[self.Fields.SIN] = ["None"]
        return d

    @lru_cache(maxsize=1)
    def observation_space(self):
        """Describes observation space for a single scalar continuous value

        Returns:
            gym.spaces.Space -- The rllib policy observation space
        """
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.COS] = gym.spaces.Box(-1.0, 1.0, shape=(1, ), dtype=np.float32)
        d.spaces[self.Fields.SIN] = gym.spaces.Box(-1.0, 1.0, shape=(1, ), dtype=np.float32)

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
        unit = self.config.wrapped.observation_units()[self.config.observation][self.config.index]
        angle = self.config.wrapped.get_observation(other_obs, obs_space, obs_units)[self.config.observation][self.config.index]
        angle_rad = Convert(angle, unit, "rad")

        d[self.Fields.COS] = np.asarray([math.cos(angle_rad)], dtype=np.float32)
        d[self.Fields.SIN] = np.asarray([math.sin(angle_rad)], dtype=np.float32)
        return d

    def action_space(self):
        """No Actions
        """
        return None

    def apply_action(self, action, observation, action_space, obs_space, obs_units):
        """No Actions
        """
        return None
