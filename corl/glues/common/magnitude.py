"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Wrapper glue which returns the magnitude of its wrapped glue as an observation.

Author: Jamie Cunningham
"""
import typing
from collections import OrderedDict
from functools import lru_cache

import gym
import numpy as np

from corl.glues.base_wrapper import BaseWrapperGlue, BaseWrapperGlueValidator
from corl.glues.common.observe_sensor import ObserveSensor


class MagnitudeGlueValidator(BaseWrapperGlueValidator):
    """
    wrapped_glue: ObserveSensor
        wrapped observe sensor
    """
    wrapped: ObserveSensor


class MagnitudeGlue(BaseWrapperGlue):
    """
    Returns the magnitude of the wrapped glue's returned observation.
    """

    class Fields:
        """
        field data
        """
        MAG = "mag"

    def __init__(self, **kwargs):
        self.config: MagnitudeGlueValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[MagnitudeGlueValidator]:
        """returns the validator for this class

        Returns:
            MagnitudeGlueValidator -- A pydantic validator to be used to validate kwargs
        """
        return MagnitudeGlueValidator

    @lru_cache(maxsize=1)
    def get_unique_name(self) -> str:
        return self.glue().get_unique_name() + "_Magnitude"

    @lru_cache(maxsize=1)
    def observation_space(self):
        tmp = self.glue().observation_space()
        if not tmp:
            raise RecursionError("the glue the magnitutde glue is wrapping does not have a valid obs space")
        obs_space = tmp[self.config.wrapped.Fields.DIRECT_OBSERVATION]
        d = gym.spaces.dict.Dict()
        d.spaces[
            self.Fields.MAG
        ] = gym.spaces.Box(0, np.maximum(np.linalg.norm(obs_space.low), np.linalg.norm(obs_space.high)), shape=(1, ), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units):
        obs = self.glue().get_observation(other_obs, obs_space, obs_units)
        if isinstance(obs, dict):
            obs = obs[self.config.wrapped.Fields.DIRECT_OBSERVATION]
        else:
            raise RuntimeError("This glue is not capable of handing wrapped obs that are not a dict")
        d = OrderedDict()
        d[self.Fields.MAG] = np.array([np.linalg.norm(obs)], dtype=np.float32)  # type: ignore
        return d

    def action_space(self):
        """No Actions
        """
        return None

    def apply_action(self, action, observation, action_space, obs_space, obs_units):
        """No Actions
        """
        return None
