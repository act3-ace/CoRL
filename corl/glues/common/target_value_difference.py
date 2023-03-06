# pylint: disable=no-member
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
import math
import typing
from collections import OrderedDict
from functools import lru_cache

import gym
import numpy as np

from corl.glues.base_dict_wrapper import BaseDictWrapperGlue, BaseDictWrapperGlueValidator
from corl.glues.common.target_value import LimitConfigValidator
from corl.libraries import units


class TargetValueDifferenceValidator(BaseDictWrapperGlueValidator):
    """
    target_value: float - Value to report the difference around
    unit: str - unit for the target value
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
    target_value: typing.Optional[float] = None
    target_value_index: typing.Optional[int] = 0
    target_value_field: typing.Optional[str] = "direct_observation"
    unit: str
    value_index: int = 0
    is_wrap: bool = False
    is_rad: bool = False
    method: int = 0
    is_abs: bool = False
    limit: LimitConfigValidator
    unique_name: str = ''
    remove_invalid_obs: bool = False
    always_valid: bool = False


class TargetValueDifference(BaseDictWrapperGlue):
    """
    Glue class for an observation of some constant value
    """
    SENSOR_STR = "sensor"
    TARGET_STR = "target"

    class Fields:  # pylint: disable=too-few-public-methods
        """
        field data
        """
        TARGET_VALUE_DIFF = "target_value_diff"

    def __init__(self, **kwargs) -> None:
        self.config: TargetValueDifferenceValidator
        super().__init__(**kwargs)
        keys = self.glues().keys()
        if not isinstance(self.config.target_value, float) and len(self.glues()) != 2:  # type: ignore
            raise ValueError("self.config.target_value not defined - expecting two glues to get target from 2nd glue")

        if not isinstance(self.config.target_value, float) and len(self.glues()) == 2:  # type: ignore
            if TargetValueDifference.TARGET_STR not in keys:
                raise KeyError(f"Expecting to see {TargetValueDifference.TARGET_STR} in keys - {keys}")

        if isinstance(self.config.target_value, float) and len(self.glues()) != 1:  # type: ignore
            raise ValueError("self.config.target_value defined - expecting one glue")

        if TargetValueDifference.SENSOR_STR not in keys:
            raise KeyError(f"Expecting to see {TargetValueDifference.SENSOR_STR} in keys - {keys}")

        self._logger = logging.getLogger(TargetValueDifference.__name__)

    @property
    def get_validator(self) -> typing.Type[TargetValueDifferenceValidator]:
        """Return validator"""
        return TargetValueDifferenceValidator

    @lru_cache(maxsize=1)
    def get_unique_name(self) -> str:
        """Class method that retrieves the unique name for the glue instance
        """
        unique_name = self.config.unique_name  # type: ignore
        if unique_name:
            return unique_name

        wrapped_glue_name = self.glues()[TargetValueDifference.SENSOR_STR].get_unique_name()
        if wrapped_glue_name is None:
            return None
        return wrapped_glue_name + "Diff"

    def invalid_value(self) -> OrderedDict:
        """When invalid return a value of 0

        TODO: this may need to be self.min in the case that the minimum is larger than 0 (i.e. a harddeck)

        Returns:
            OrderedDict -- Dictionary with <FIELD> entry containing 1D array
        """
        d = OrderedDict()

        if isinstance(self.config.target_value, float):  # type: ignore
            target_value = self.config.target_value  # type: ignore
        else:
            target_value = self.glues()[TargetValueDifference.TARGET_STR].get_observation({}, {}, {})  # type: ignore

        d[self.Fields.TARGET_VALUE] = np.asarray([target_value], dtype=np.float32)  # type: ignore
        return d

    @lru_cache(maxsize=1)
    def observation_space(self):
        """Observation space"""
        space = self.glues()[TargetValueDifference.SENSOR_STR].observation_space()

        d = gym.spaces.dict.Dict()
        if len(space.spaces) > 1:
            raise RuntimeError("Target value difference can only wrap a glue with one output")

        for field in space.spaces:
            d.spaces[field + "_diff"] = gym.spaces.Box(self.config.limit.minimum, self.config.limit.maximum, shape=(1, ), dtype=np.float32)
            if not self.config.remove_invalid_obs:
                d.spaces[field + "_diff_invalid"] = gym.spaces.Discrete(2)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units):  # pylint: disable=too-many-statements
        """Get observation"""

        def get_wrap_diff(A, B):
            """Returns the min diff angle

             RAD A	         Deg A   RAD A	       Deg B   abs_diff_mod_360	Diff
              1.047197551	  60     2.094395102    120	    60	             60
              2.094395102	 120 	 1.047197551     60	    60	             60
             -2.094395102	-120	-1.047197551    -60	    60	             60
              6.108652382	 350	 0.1745329252    10	   340	             20
             -2.967059728	-170	 2.967059728    170	   340	             20

            Arguments:
                A {float} -- Angle 1 - rad or deg
                B {float} -- Angle 2 - rad or deg

            Returns:
                float -- the min diff angle
            """

            # Convert to degrees if needed.
            temp_a = math.degrees(A) if self.config.is_rad else A
            temp_b = math.degrees(B) if self.config.is_rad else B

            if self.config.method == 1:
                # Compute the diff as abs min angle (always positive)
                abs_diff_mod_360 = abs(temp_a - temp_b) % 360
                result = 360 - abs_diff_mod_360 if abs_diff_mod_360 > 180 else abs_diff_mod_360
            else:
                # Compute the diff as min angle maintain sign for direction
                diff = temp_a - temp_b
                result = (diff + 180) % 360 - 180

            # Return the diff in deg if radians if needed.
            return math.radians(result) if self.config.is_rad else result

        glue = self.glues()[TargetValueDifference.SENSOR_STR]
        obs = glue.get_observation(other_obs, obs_space, obs_units)

        observation_units = None
        if hasattr(glue, "observation_units"):
            observation_units = glue.observation_units()  # type: ignore

        if observation_units and self.config.unit:
            observation_unit = None
            length = 0
            for item in observation_units:
                observation_unit = observation_units[item][0]
                length += 1

            if length != 1:
                raise RuntimeError(f"observation_units not of length 1: {observation_units}")

            if units.GetUnitFromStr(observation_unit) != units.GetUnitFromStr(self.config.unit):
                raise RuntimeError(f"Target value units [{self.config.unit}] and observation units [{observation_unit}] don't match")

        target_invalid = False
        target_name = " "
        if isinstance(self.config.target_value, float):
            target_value = self.config.target_value
        else:
            # TODO make this more configurable... assumes 1 value for target
            if hasattr(self.glues()[TargetValueDifference.TARGET_STR], "_sensor"):  # type: ignore
                target_invalid = not self.glues()[  # pylint: disable=protected-access  # type: ignore
                    TargetValueDifference.TARGET_STR]._sensor.valid  # type: ignore
                target_name = self.glues()[  # pylint: disable=protected-access  # type: ignore
                    TargetValueDifference.TARGET_STR]._sensor_name  # type: ignore
            target_value = self.glues()[TargetValueDifference.TARGET_STR].get_observation(other_obs, obs_space, obs_units)[
                self.config.target_value_field][  # type: ignore
                    self.config.target_value_index]

            # I would think that this would be needed by the target value but is already converted.
            # if units.GetUnitFromStr(self.glues()[1]._sensor._properties.unit[0]) != units.GetUnitFromStr(self.config.unit):
            #     target_value = units.Convert( target_value,
            #                                   units.GetUnitFromStr(self.glues()[1]._sensor._properties.unit[0]),
            #                                   units.GetUnitFromStr(self.config.unit))
        # disable the invalid check and always treat the part as valid
        if self.config.always_valid:
            target_invalid = False
        d = OrderedDict()
        if not isinstance(obs, dict):
            raise RuntimeError("This glue cannot handle a wrapped glue outputting something other than a dict")

        if target_invalid:
            for field, value in obs.items():
                d[field + "_diff"] = np.array([self.config.limit.minimum], dtype=np.float32)
                if not self.config.remove_invalid_obs:
                    d[field + "_diff_invalid"] = target_invalid  # type: ignore
        else:
            for field, value in obs.items():
                if self.config.is_wrap:
                    # Note: this always returns the min angle diff and is positive
                    diff = get_wrap_diff(target_value, value[self.config.value_index])
                else:
                    diff = target_value - value[self.config.value_index]

                if self.config.is_abs:
                    diff = abs(diff)

                self._logger.debug(f"{TargetValueDifference.__name__}::{target_name}::obs = {value[self.config.value_index]}")
                self._logger.debug(f"{TargetValueDifference.__name__}::{target_name}::target = {target_value}")
                self._logger.debug(f"{TargetValueDifference.__name__}::{target_name}::diff = {diff}")

                d[field + "_diff"] = np.array([diff], dtype=np.float32)
                if not self.config.remove_invalid_obs:
                    d[field + "_diff_invalid"] = glue.agent_removed()  # type: ignore

        return d

    @lru_cache(maxsize=1)
    def action_space(self):
        """Action space"""
        return None

    def apply_action(self, action, observation, action_space, obs_space, obs_units):  # pylint: disable=unused-argument
        """Apply action"""
        return None
