"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Sensors for GymnasiumSimulator
"""
import typing
from collections import OrderedDict
from functools import cached_property

import numpy as np

from corl.glues.base_dict_wrapper import BaseDictWrapperGlue
from corl.libraries.property import BoxProp, DictProp


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

        if "quantity" not in self.glues():
            raise KeyError("Missing key: quantity")

        wrapped_name = self.glues()["quantity"].get_unique_name()

        self._uname: str = ""
        if wrapped_name is None:
            self._uname = "ProjectedQuantity"
        else:
            self._uname = f"Projected_{wrapped_name}"

    def get_unique_name(self) -> str:
        return self._uname

    @cached_property
    def observation_prop(self):
        if "quantity" not in self.glues():
            raise KeyError("Missing key: quantity")

        tmp = self.glues()["quantity"].observation_prop
        if not isinstance(tmp, DictProp):
            raise RuntimeError("projected_quantity glue received an invalid Prop from the glue it is wrapping")
        wrapped_space = next(iter(tmp.spaces.values()))
        max_wrapped_obs = np.amax(np.array([np.abs(wrapped_space.low), np.abs(wrapped_space.high)]))

        return DictProp(
            spaces={self.Fields.PROJECTED_QUANTITY: BoxProp(low=[-max_wrapped_obs], high=[max_wrapped_obs], unit=wrapped_space.get_units())}
        )

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> typing.OrderedDict[str, np.ndarray]:
        d = OrderedDict()

        observations = {
            k: next(iter(v.get_observation(other_obs, obs_space, obs_units).values())).m for k, v in self.glues().items()  # type: ignore
        }

        projected_value = observations.pop("quantity")
        projected_value *= np.prod(np.cos(list(observations.values())))

        d[self.Fields.PROJECTED_QUANTITY] = self.observation_prop[self.Fields.PROJECTED_QUANTITY].create_quantity(projected_value)
        return d
