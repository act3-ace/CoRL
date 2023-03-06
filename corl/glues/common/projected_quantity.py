"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Sensors for OpenAIGymSimulator
"""
import typing
from collections import OrderedDict
from functools import lru_cache

import gym
import numpy as np

from corl.glues.base_dict_wrapper import BaseDictWrapperGlue


class ProjectedQuantity(BaseDictWrapperGlue):
    """
    Glue that takes a quantity and a number of angles and projects the quantity in the direction of the angles given.
    """

    class Fields:
        """
        Fields in this glue
        """

        PROJECTED_QUANTITY = "projected_quantity"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if 'quantity' not in self.glues().keys():
            raise KeyError('Missing key: quantity')

    @lru_cache(maxsize=1)
    def get_unique_name(self) -> str:
        wrapped_name = self.glues()["quantity"].get_unique_name()

        if wrapped_name is None:
            return "ProjectedQuantity"
        return "Projected_" + wrapped_name

    @lru_cache(maxsize=1)
    def observation_space(self):
        tmp = self.glues()["quantity"].observation_space()
        if not isinstance(tmp, gym.spaces.Dict):
            raise RuntimeError("projected_quantity glue recieved an invalid observation from the glue it is wrapping")
        wrapped_space = list(tmp.spaces.values())[0]
        max_wrapped_obs = np.amax(np.array([np.abs(wrapped_space.low), np.abs(wrapped_space.high)]))

        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.PROJECTED_QUANTITY] = gym.spaces.Box(-max_wrapped_obs, max_wrapped_obs, shape=(1, ), dtype=np.float32)

        return d

    @lru_cache(maxsize=1)
    def observation_units(self):
        """
        Returns Dict space with the units of the 'quantity' glue
        """
        d = gym.spaces.dict.Dict()
        quantity_glue = self.glues()['quantity']
        try:
            d.spaces[self.Fields.PROJECTED_QUANTITY] = quantity_glue.config.output_units
        except AttributeError:
            d.spaces[self.Fields.PROJECTED_QUANTITY] = quantity_glue.observation_units()
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> typing.OrderedDict[str, np.ndarray]:
        d = OrderedDict()

        observations = {
            k: list(v.get_observation(other_obs, obs_space, obs_units).values())[0]  # type: ignore
            for k, v in self.glues().items()
        }  # type: ignore[union-attr]

        projected_value = observations.pop('quantity')
        projected_value *= np.prod(np.cos(list(observations.values())))

        d[self.Fields.PROJECTED_QUANTITY] = projected_value

        return d

    def action_space(self):
        ...

    def apply_action(self, action, observation, action_space, obs_space, obs_units):
        ...
