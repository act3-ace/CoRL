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
from collections import OrderedDict
from functools import cached_property

import gymnasium
import numpy as np

from corl.glues.base_wrapper import BaseWrapperGlue, BaseWrapperGlueValidator
from corl.glues.common.observe_sensor import ObserveSensor
from corl.libraries.property import BoxProp, DictProp
from corl.libraries.units import corl_get_ureg


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

    def __init__(self, **kwargs) -> None:
        self.config: TrigValuesValidator
        super().__init__(**kwargs)

        self._uname = f"{self.config.name}"

    @staticmethod
    def get_validator() -> type[TrigValuesValidator]:
        return TrigValuesValidator

    def get_unique_name(self):
        """Provides a unique name of the glue to differentiate  it from other glues."""
        return self._uname

    @cached_property
    def observation_prop(self):
        return DictProp(
            spaces={
                self.Fields.COS: BoxProp(low=[-1.0], high=[1.0], dtype=np.dtype("float32"), unit="dimensionless"),
                self.Fields.SIN: BoxProp(low=[-1.0], high=[1.0], dtype=np.dtype("float32"), unit="dimensionless"),
            }
        )

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> OrderedDict:
        """Constructs a portion of the observation space

        Arguments:
            platforms {List[BasePlatform]} -- List of current platforms.
                Guaranteed to be length of 1 and set to self.obs_agent

        Returns:
            OrderedDict -- Dictionary containing observation in <FIELD> key
        """
        d = OrderedDict()
        angle_rad = self.config.wrapped.get_observation(other_obs, obs_space, obs_units)[self.config.observation][self.config.index].m_as(
            "rad"
        )

        _Quantity = corl_get_ureg().Quantity
        d[self.Fields.COS] = _Quantity(np.asarray([math.cos(angle_rad)], dtype=np.float32), "dimensionless")
        d[self.Fields.SIN] = _Quantity(np.asarray([math.sin(angle_rad)], dtype=np.float32), "dimensionless")
        return d

    @cached_property
    def action_space(self) -> gymnasium.spaces.Space | None:
        """No Actions"""
        return None

    def apply_action(self, action, observation, action_space, obs_space, obs_units):  # noqa: PLR6301
        """No Actions"""
        return
