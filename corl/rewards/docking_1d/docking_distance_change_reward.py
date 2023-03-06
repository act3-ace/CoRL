"""
This module implements the Reward Functions and Reward Validators specific to the 1D Docking task.
"""

from collections import OrderedDict

import numpy as np
from numpy_ringbuffer import RingBuffer

from corl.libraries.environment_dict import RewardDict
from corl.libraries.state_dict import StateDict
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from corl.simulators.common_platform_utils import get_platform_by_name, get_sensor_by_name


class DockingDistanceChangeRewardValidator(RewardFuncBaseValidator):
    """
    scale: Scalar value to adjust magnitude of the reward
    """

    scale: float = 1.0
    position_sensor_name: str


class DockingDistanceChangeReward(RewardFuncBase):
    """
    This RewardFuncBase extension is responsible for calculating the reward associated with a change in agent position.
    """

    def __init__(self, **kwargs):
        self.config: DockingDistanceChangeRewardValidator
        super().__init__(**kwargs)
        self._dist_buffer = RingBuffer(capacity=2, dtype=float)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return DockingDistanceChangeRewardValidator

    def __call__(
        self,
        observation: OrderedDict,
        action,
        next_observation: OrderedDict,
        state: StateDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> RewardDict:
        """
        This method calculates the current position of the agent and compares it to the previous position. The
        difference is used to return a proportional reward.

        Parameters
        ----------
        observation : OrderedDict
            The observations available to the agent from the previous state.
        action
            The last action performed by the agent.
        next_observation : OrderedDict
            The observations available to the agent from the current state.
        state : StateDict
            The previous state of the simulation.
        next_state : StateDict
            The current state of the simulation.
        observation_space : StateDict
            The agent's observation space.
        observation_units : StateDict
            The units corresponding to values in the observation_space?

        Returns
        -------
        reward : RewardDict
            The agent's reward for their change in distance.
        """

        reward = RewardDict()
        val = 0

        deputy = get_platform_by_name(next_state, self.config.platform_names[0])
        position_sensor = get_sensor_by_name(deputy, self.config.position_sensor_name)  # type: ignore
        deputy_position = position_sensor.get_measurement()
        chief_position = np.array([0])  # hardcoded to origin

        distance = abs(chief_position - deputy_position)
        self._dist_buffer.append(distance[0])

        if len(self._dist_buffer) == 2:
            val = self.config.scale * (self._dist_buffer[0] - self._dist_buffer[1])

        reward[self.config.agent_name] = val

        return reward
