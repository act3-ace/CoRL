"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from collections import OrderedDict
from functools import cached_property

import numpy as np
from gymnasium.spaces import Dict
from pydantic import BaseModel, model_validator

from corl.glues.base_glue import BaseAgentPlatformGlue, BaseAgentPlatformGlueValidator
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.property import BoxProp, DictProp
from corl.libraries.units import Quantity, corl_get_ureg, corl_quantity


class LimitConfigValidator(BaseModel):
    """
    minimum: defines the min the obs of the glue may produce
    maximum: defines the max the obs of the glue may produce
    unit: defines the unit for the limits
    clip: defines if the glue should clip the obs values to between min/max
    """

    minimum: Quantity
    maximum: Quantity
    unit: str = "dimensionless"

    @model_validator(mode="before")
    @classmethod
    def check_units(cls, values):
        """
        This function will attmempt to align all of the
        units between the different units that can be
        specified in this validator
        """
        # if the user does not pass a quantity, we attempt to either make a quantity out of it or let it blow up
        if isinstance(values["minimum"], int | float):
            min_val = corl_quantity()(values["minimum"], values.get("unit", "dimensionless"))
        else:
            min_val = values["minimum"]

        if isinstance(values["maximum"], int | float):
            max_val = corl_quantity()(values["maximum"], values.get("unit", "dimensionless"))
        else:
            max_val = values["maximum"]

        # if the limit validator is dimensionless, we will assume min and max are the same unit
        if values.get("unit", "dimensionless") == "dimensionless":
            if min_val.u != max_val.u:
                raise ValueError(f"LimitConfigValidator did not receive a unit, but {min_val=} does not have the same unitas {max_val=}")

            values["unit"] = str(min_val.u)
        # if the limit validator has a unit, we will convert min and max to that
        min_is_comp = min_val.is_compatible_with(values["unit"])
        max_is_comp = max_val.is_compatible_with(values["unit"])

        if not (min_is_comp and max_is_comp):
            raise ValueError(
                f"LimitConfigValidator was told it's limits were dimension {values['unit']}",
                f"but your unit registry does not believe {min_val=} or {max_val=} are compatible"
                f"with this unit type.  {min_is_comp=}, {max_is_comp=}",
            )

        values["minimum"] = min_val.to(values["unit"])
        values["maximum"] = max_val.to(values["unit"])
        return values


class TargetValueValidator(BaseAgentPlatformGlueValidator):
    """
    target_value: the Value this glue should produce this episode
    unit: the unit of the value produced
    """

    target_value: Quantity
    limit: LimitConfigValidator

    @model_validator(mode="after")
    def check_tv_units(self):
        """
        will convert the target value units to the units of the limit
        """
        try:
            self.target_value = self.target_value.to(self.limit.unit)
        except Exception as exc:  # noqa
            raise ValueError(f"TargetValueValidator Tried to convert {self.target_value=} to {self.limit.unit=}, but could not") from exc
        return self


class TargetValue(BaseAgentPlatformGlue):
    """
    Glue class for an observation of some constant value
    """

    class Fields:
        """
        field data
        """

        TARGET_VALUE = "target_value"

    def __init__(self, **kwargs) -> None:
        self.config: TargetValueValidator
        super().__init__(**kwargs)

        self._uname = f"{self.config.name}TargetValue"

        self._target_value = OrderedDict()
        tmp = corl_get_ureg().Quantity(np.asarray([self.config.target_value.m], np.float32), self.config.limit.unit)
        assert isinstance(self.observation_space, Dict)
        tmp = EnvSpaceUtil.clip_space_sample_to_space(tmp, self.observation_space[self.Fields.TARGET_VALUE])
        self._target_value[self.Fields.TARGET_VALUE] = tmp

    @staticmethod
    def get_validator() -> type[TargetValueValidator]:
        return TargetValueValidator

    def get_unique_name(self):
        """Provides a unique name of the glue to differentiate it from other glues."""
        return self._uname

    def invalid_value(self) -> OrderedDict:
        """When invalid return a value of 0

        TODO: this may need to be self.min in the case that the minimum is larger than 0 (i.e. a harddeck)

        Returns:
            OrderedDict -- Dictionary with <FIELD> entry containing 1D array
        """
        return self._target_value

    @cached_property
    def observation_prop(self):
        # DictProp, BoxProp
        return DictProp(
            spaces={
                self.Fields.TARGET_VALUE: BoxProp(
                    low=[self.config.limit.minimum.m_as(self.config.limit.unit)],
                    high=[self.config.limit.maximum.m_as(self.config.limit.unit)],
                    dtype=np.dtype(np.float32),
                    unit=self.config.limit.unit,
                )
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

        return self._target_value
