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

import numpy as np

from corl.libraries.utils import get_wrap_diff
from corl.rewards.base_measurement_operation import BaseMeasurementOperation, BaseMeasurementOperationValidator


class ExponentialDecayFromTargetValueValidator(BaseMeasurementOperationValidator):
    """
    reward scale: scale of this reward, this would be the maximum reward value for
                  a given time step
    eps: the length of the reward curve for the exponential decay, would recommend playing with
        this value in a plotting software to determine the value you need
    target_value: the value with which to take the difference from the wrapped observation value
    index: the index with which to pull data out of the observation extractor, useful if len(observation) > 1
    is_wrap: if the obs difference needs to wrap around 0/360
    is_rad: if the obs difference is in terms of radians
    closer: to only reward with this reward if the difference to the target value
            is less than the last time step
    closer_tolerance: the difference tolerance at which point the agent is close enough
                      to the target value that "closer" is not a concern
    """

    reward_scale: float
    eps: float
    target_value: float = 0
    index: int | None = 0
    is_wrap: bool = False
    is_rad: bool = False
    method: bool = False
    closer: bool = False
    closer_tolerance: float = 0.0


class ExponentialDecayFromTargetValue(BaseMeasurementOperation):
    """
    Exponential Decay from Target Value
    wraps some sort of observation and takes in a target value,
    the reward based on the difference between the target value
    and the observation. the reward exponentially decays in value
    the further you are away from the target value. causing a spike at the target value
    """

    @staticmethod
    def get_validator() -> type[ExponentialDecayFromTargetValueValidator]:
        return ExponentialDecayFromTargetValueValidator

    def __init__(self, **kwargs) -> None:
        self.config: ExponentialDecayFromTargetValueValidator
        super().__init__(**kwargs)
        self._last_value = None
        self._logger = logging.getLogger(self.name)

    def __call__(
        self,
        observation,
        action,
        next_observation,
        state,
        next_state,
        observation_space,
        observation_units,
    ) -> float:
        reward = 0.0
        if self.config.agent_name not in next_observation:
            return reward

        obs = self.extractor.value(next_observation[self.config.agent_name])[self.config.index].m

        if self.config.is_wrap:
            # Note: this always returns the min angle diff and is positive
            diff = get_wrap_diff(obs, self.config.target_value, self.config.is_rad, self.config.method)
        else:
            diff = obs - self.config.target_value

        abs_diff = abs(diff)
        if self._last_value is None:
            self._last_value = abs_diff  # type: ignore

        func_applied: float = 0.0
        if not self.config.closer or ((self._last_value >= abs_diff) or abs_diff < self.config.closer_tolerance):  # type: ignore
            func_applied = np.exp(-np.abs(diff / self.config.eps))

        reward = self.config.reward_scale * func_applied

        self._last_value = abs_diff  # type: ignore
        return reward
