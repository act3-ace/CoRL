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

import numpy as np

from corl.libraries.environment_dict import RewardDict
from corl.rewards.base_measurement_operation import BaseMeasurementOperation, BaseMeasurementOperationValidator


class ExponentialDecayFromTargetValueValidator(BaseMeasurementOperationValidator):
    """
    reward scale: scale of this reward, this would be the maximum reward value for
                  a given timestep
    eps: the length of the reward curve for the exponential decay, would recommend playing with
        this value in a plotting software to determine the value you need
    target_value: the value with which to take the difference from the wrapped observation value
    index: the index with which to pull data out of the observation extractor, useful if len(observation) > 1
    is_wrap: if the obs difference needs to wrap around 0/360
    is_rad: if the obs difference is in terms of radians
    closer: to only reward with this reward if the difference to the target value
            is less than the last timestep
    closer_tolerance: the difference tolerance at which point the agent is close enough
                      to the target value that "closer" is not a concern
    """
    reward_scale: float
    eps: float
    target_value: typing.Optional[float] = 0
    index: typing.Optional[int] = 0
    is_wrap: typing.Optional[bool] = False
    is_rad: typing.Optional[bool] = False
    method: typing.Optional[bool] = False
    closer: typing.Optional[bool] = False
    closer_tolerance: typing.Optional[float] = 0.0


class ExponentialDecayFromTargetValue(BaseMeasurementOperation):
    """
    Exponential Decay from Target Value
    wraps some sort of observation and takes in a target value,
    the reward based on the difference between the target value
    and the observation. the reward exponentially decays in value
    the further you are away from the target value. causing a spike at the target value
    """

    @property
    def get_validator(self) -> typing.Type[ExponentialDecayFromTargetValueValidator]:
        return ExponentialDecayFromTargetValueValidator

    def __init__(self, **kwargs) -> None:
        self.config: ExponentialDecayFromTargetValueValidator
        super().__init__(**kwargs)
        self._last_value = None
        self._logger = logging.getLogger(self.name)

    def __call__(self, observation, action, next_observation, state, next_state, observation_space, observation_units) -> RewardDict:

        reward = RewardDict()
        reward[self.config.agent_name] = 0

        if self.config.agent_name not in next_observation:
            return reward

        obs = self.extractor.value(next_observation[self.config.agent_name])[self.config.index]

        def get_wrap_diff(A, B):
            """Returns the min diff angle

             RAD A	         Deg A   RAD A	       Deg B   abs_diff_mod_360	M Diff 1 Diff A-B	M 2 Diff
              1.047197551	  60     2.094395102    120	    60	             60      -60        -60
              2.094395102	 120 	 1.047197551     60	    60	             60       60         60
             -2.094395102	-120	-1.047197551    -60	    60	             60      -60        -60
              6.108652382	 350	 0.1745329252    10	   340	             20      340        -20
             -2.967059728	-170	 2.967059728    170	   340	             20     -340         20

            Arguments:
                A {float} -- Angle 1 - rad or deg
                B {float} -- Angle 2 - rad or deg

            Returns:
                float -- the min diff angle
            """

            # Convert to degrees if needed.
            temp_a = math.degrees(A) if self.config.is_rad else A
            temp_b = math.degrees(B) if self.config.is_rad else B

            if self.config.method:
                # Compute the diff as abs min angle (always positive)
                abs_diff_mod_360 = abs(temp_a - temp_b) % 360
                result = 360 - abs_diff_mod_360 if abs_diff_mod_360 > 180 else abs_diff_mod_360
            else:
                # Compute the diff as min angle maintain sign for direction
                diff = temp_a - temp_b
                result = (diff + 180) % 360 - 180

            # Return the diff in deg if radians if needed.
            return math.radians(result) if self.config.is_rad else result

        if self.config.is_wrap:
            # Note: this always returns the min angle diff and is positive
            diff = get_wrap_diff(obs, self.config.target_value)
        else:
            diff = obs - self.config.target_value

        abs_diff = abs(diff)
        if self._last_value is None:
            self._last_value = abs_diff

        func_applied = 0
        if not self.config.closer or ((self._last_value >= abs_diff) or abs_diff < self.config.closer_tolerance):
            func_applied = np.exp(-np.abs(diff / self.config.eps))

        reward[self.config.agent_name] = self.config.reward_scale * func_applied

        self._last_value = abs_diff
        return reward
