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

from collections import OrderedDict
from functools import cached_property

import gymnasium
import numpy as np

from corl.glues.base_wrapper import BaseWrapperGlue, BaseWrapperGlueValidator
from corl.glues.common.observe_sensor import ObserveSensor
from corl.libraries.property import BoxProp, DictProp
from corl.libraries.units import corl_quantity


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

    def __init__(self, **kwargs) -> None:
        self.config: UnitVectorGlueValidator
        super().__init__(**kwargs)

        self._uname = f"{self.glue().get_unique_name()}_UnitVector"

    @staticmethod
    def get_validator() -> type[UnitVectorGlueValidator]:
        return UnitVectorGlueValidator

    def get_unique_name(self):
        return self._uname

    @cached_property
    def observation_prop(self):
        tmp = self.glue().observation_space[self.config.wrapped.Fields.DIRECT_OBSERVATION].shape
        return DictProp(
            spaces={
                self.Fields.UNIT_VEC: BoxProp(
                    low=[-1.0] * tmp[0], high=[1.0] * tmp[0], shape=tmp, dtype=np.dtype("float32"), unit="dimensionless"
                )
            }
        )

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> OrderedDict:
        direct_obs = self.config.wrapped.get_observation(other_obs, obs_space, obs_units)[self.config.wrapped.Fields.DIRECT_OBSERVATION]

        # Compute unit vector
        norm = np.linalg.norm(direct_obs.m)
        unit_vec = direct_obs.m / norm

        # Return observation
        d = OrderedDict()
        d[self.Fields.UNIT_VEC] = corl_quantity()(np.array(unit_vec, dtype=np.float32), "dimensionless")
        return d

    @cached_property
    def action_space(self) -> gymnasium.spaces.Space | None:
        """No Actions"""
        return None

    def apply_action(self, action, observation, action_space, obs_space, obs_units):  # noqa: PLR6301
        """No Actions"""
        return
