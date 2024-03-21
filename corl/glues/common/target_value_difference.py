"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import logging
from collections import OrderedDict
from functools import cached_property

import numpy as np

from corl.glues.base_dict_wrapper import BaseDictWrapperGlue, BaseDictWrapperGlueValidator
from corl.glues.common.target_value import LimitConfigValidator
from corl.libraries.property import BoxProp, DictProp
from corl.libraries.units import Quantity, corl_quantity
from corl.libraries.utils import get_wrap_diff_quant


class TargetValueDifferenceValidator(BaseDictWrapperGlueValidator):
    """
    target_value: Quantity - Value to report the difference around
    value_index: - index to extract the value from an ndarray extractor
    is_wrap: bool - if the target value difference needs to wrap around the limits
    is_rad: bool - if the read observation is in radians
    method: int - TODO: fill this is
    is_abs: bool - if the difference to the target value should be reported as absolute value
    limit: LimitConfigValidator - limit configuration
    unique_name: str - optional unique name for a target value difference glue
    remove_invalid_obs: if the invalid flags should be removed from the obs
    always_valid: never output a fixed value based on a check
    that an underlying glue may have a part that is invalid
    """

    target_value: Quantity | None = None
    target_value_index: int | None = 0
    target_value_field: str | None = "direct_observation"
    value_index: int = 0
    is_wrap: bool = False
    is_rad: bool = False
    method: int = 0
    is_abs: bool = False
    limit: LimitConfigValidator
    unique_name: str = ""
    always_valid: bool = False


class TargetValueDifference(BaseDictWrapperGlue):
    """
    Glue class for an observation of some constant value
    """

    SENSOR_STR = "sensor"
    TARGET_STR = "target"

    class Fields:
        """
        field data
        """

        TARGET_VALUE_DIFF = "diff"

    def __init__(self, **kwargs) -> None:
        self.config: TargetValueDifferenceValidator
        super().__init__(**kwargs)
        keys = self.glues().keys()
        if not isinstance(self.config.target_value, Quantity) and len(self.glues()) != 2:
            raise ValueError("self.config.target_value not defined - expecting two glues to get target from 2nd glue")

        if not isinstance(self.config.target_value, Quantity) and len(self.glues()) == 2 and TargetValueDifference.TARGET_STR not in keys:
            raise KeyError(f"Expecting to see {TargetValueDifference.TARGET_STR} in keys - {keys}")

        if isinstance(self.config.target_value, Quantity) and len(self.glues()) != 1:
            raise ValueError("self.config.target_value defined - expecting one glue")

        if TargetValueDifference.SENSOR_STR not in keys:
            raise KeyError(f"Expecting to see {TargetValueDifference.SENSOR_STR} in keys - {keys}")

        self._logger = logging.getLogger(TargetValueDifference.__name__)

        if unique_name := self.config.unique_name:
            self._uname = unique_name
        else:
            wrapped_glue_name = self.glues()[TargetValueDifference.SENSOR_STR].get_unique_name()
            self._uname = None if wrapped_glue_name is None else f"{wrapped_glue_name}Diff"

    @staticmethod
    def get_validator() -> type[TargetValueDifferenceValidator]:
        """Return validator"""
        return TargetValueDifferenceValidator

    def get_unique_name(self) -> str:
        """Class method that retrieves the unique name for the glue instance"""
        return self._uname

    @cached_property
    def observation_prop(self):
        space = self.glues()[TargetValueDifference.SENSOR_STR].observation_space

        if len(space.spaces) > 1:
            raise RuntimeError("Target value difference can only wrap a glue with one output")

        new_field_name = self.Fields.TARGET_VALUE_DIFF
        for field in space.spaces:
            new_field_name = f"{field}_diff"
        return DictProp(
            spaces={
                new_field_name: BoxProp(
                    low=self.config.limit.minimum.m,
                    high=self.config.limit.maximum.m,
                    dtype=np.dtype("float32"),
                    unit=self.config.limit.unit,
                )
            }
        )

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units):
        """Get observation"""
        glue = self.glues()[TargetValueDifference.SENSOR_STR]
        obs = glue.get_observation(other_obs, obs_space, obs_units)
        target_invalid = False

        if isinstance(self.config.target_value, Quantity):
            target_value = self.config.target_value
            if np.isscalar(target_value.m):
                target_value = corl_quantity()(np.asarray([target_value.m], np.float32), self.config.target_value.u)
        else:
            # TODO make this more configurable... assumes 1 value for target
            if hasattr(self.glues()[TargetValueDifference.TARGET_STR], "_sensor"):
                target_invalid = not self.glues()[TargetValueDifference.TARGET_STR]._sensor.valid  # type: ignore # noqa: SLF001
            target_value = self.glues()[TargetValueDifference.TARGET_STR].get_observation(other_obs, obs_space, obs_units)[
                self.config.target_value_field
            ][  # type: ignore
                self.config.target_value_index
            ]

        assert isinstance(target_value, Quantity)
        target_value = target_value.to(self.config.limit.unit)

        if self.config.always_valid:
            target_invalid = False
        d = OrderedDict()

        if not isinstance(obs, dict):
            raise RuntimeError("This glue cannot handle a wrapped glue outputting something other than a dict")

        if target_invalid:
            for field in obs:
                d[f"{field}_diff"] = corl_quantity()(np.array([self.config.limit.minimum.m], dtype=np.float32), self.config.limit.unit)
        else:
            for field, value in obs.items():
                new_val = value[self.config.target_value_index] if value.m.shape else value
                diff = (
                    get_wrap_diff_quant(target_value, new_val, self.config.method)
                    if self.config.is_wrap
                    else corl_quantity()(np.asarray(target_value.m - new_val.m), target_value.u)
                )
                unit_name = self.observation_prop.spaces[f"{field}_diff"].get_units()
                diff = diff.to(unit_name)

                if self.config.is_abs:
                    diff = abs(diff.m)

                self._logger.debug(f"{TargetValueDifference.__name__}::obs = {value}")
                self._logger.debug(f"{TargetValueDifference.__name__}::target = {target_value}")
                self._logger.debug(f"{TargetValueDifference.__name__}::diff = {diff}")

                d[f"{field}_diff"] = corl_quantity()(np.asarray(diff.m.astype(np.float32)), unit_name)

        return d
