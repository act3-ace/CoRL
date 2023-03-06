"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Glue which transforms observation into a unit vector and allows custom normalization.

Author: Jamie Cunningham
"""

import typing
from collections import OrderedDict
from functools import lru_cache

import gym.spaces
import numpy as np

from corl.glues.base_wrapper import BaseWrapperGlue, BaseWrapperGlueValidator
from corl.glues.common.observe_sensor import ObserveSensor


class UnitVectorGlueValidator(BaseWrapperGlueValidator):
    """
    Validator for UnitVector glue
    """
    wrapped: ObserveSensor


class UnitVectorGlue(BaseWrapperGlue):
    """
    Transforms an observation into a unit vector.
    """

    class Fields:
        """
        field data
        """
        UNIT_VEC = "unit_vec"

    def __init__(self, **kwargs):
        self.config: UnitVectorGlueValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[UnitVectorGlueValidator]:
        return UnitVectorGlueValidator

    @lru_cache(maxsize=1)
    def get_unique_name(self) -> str:
        return self.glue().get_unique_name() + "_UnitVector"

    @lru_cache(maxsize=1)
    def observation_space(self):
        d = gym.spaces.dict.Dict()
        tmp = self.glue().observation_space()[self.config.wrapped.Fields.DIRECT_OBSERVATION].shape  # type: ignore
        d.spaces[self.Fields.UNIT_VEC] = gym.spaces.Box(-1.0, 1.0, shape=tmp, dtype=np.float32)
        return d

    @lru_cache(maxsize=1)
    def observation_units(self):
        """Return a space dictionary that indicates the units of the observation
        """
        return ["None"]

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> OrderedDict:

        direct_obs = self.config.wrapped.get_observation(other_obs, obs_space, obs_units)[self.config.wrapped.Fields.DIRECT_OBSERVATION]

        # Compute unit vector
        norm = np.linalg.norm(direct_obs)
        unit_vec = direct_obs / norm

        # Return observation
        d = OrderedDict()
        d[self.Fields.UNIT_VEC] = np.array(unit_vec, dtype=np.float32)
        return d

    def action_space(self):
        """No Actions
        """
        return None

    def apply_action(self, action, observation, action_space, obs_space, obs_units):
        """No Actions
        """
        return None
