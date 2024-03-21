"""
This module implements the Reward Functions and Reward Validators specific to the 1D Docking task.
"""

from collections import OrderedDict

import gymnasium
import numpy as np

from corl.libraries.units import Quantity
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from corl.simulators.base_simulator_state import BaseSimulatorState
from corl.simulators.common_platform_utils import get_platform_by_name, get_sensor_by_name


class DockingRewardValidator(RewardFuncBaseValidator):
    """
    This Validator ensures the DockingReward's config defines values relevant to successful and unsuccessful docking
    attempts.
    """

    success_reward: float
    timeout_reward: float
    distance_reward: float
    crash_reward: float
    timeout: float
    docking_region_radius: Quantity
    max_goal_distance: Quantity
    velocity_threshold: Quantity
    position_sensor_name: str
    velocity_sensor_name: str


class DockingReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward (or penalty) associated with a given docking attempt.
    """

    REQUIRED_UNITS = {"docking_region_radius": "meter", "velocity_threshold": "meter / second", "max_goal_distance": "meter"}

    def __init__(self, **kwargs) -> None:
        self.config: DockingRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[DockingRewardValidator]:
        """
        Method to return class's Validator.
        """
        return DockingRewardValidator

    def __call__(
        self,
        observation: OrderedDict,
        action,
        next_observation: OrderedDict,
        state: BaseSimulatorState,
        next_state: BaseSimulatorState,
        observation_space: gymnasium.Space,
        observation_units: OrderedDict,
    ) -> float:
        """
        This method determines if the agent has succeeded or failed and returns an appropriate reward.

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
            The units corresponding to values in the observation_space

        Returns
        -------
        reward
            The agent's reward for their docking attempt.
        """
        value = 0.0

        deputy = get_platform_by_name(next_state, self.config.platform_names[0])

        position_sensor = get_sensor_by_name(deputy, self.config.position_sensor_name)  # type: ignore
        velocity_sensor = get_sensor_by_name(deputy, self.config.velocity_sensor_name)  # type: ignore

        position = position_sensor.get_measurement().m
        velocity = velocity_sensor.get_measurement().m
        sim_time = deputy.sim_time  # type: ignore

        chief_position = np.array([0])
        docking_region_radius = self.config.docking_region_radius.m

        distance = abs(position - chief_position)
        in_docking = distance <= docking_region_radius

        max_velocity_exceeded = self.config.velocity_threshold.m < velocity

        if sim_time > self.config.timeout:
            # episode reached max time
            value = self.config.timeout_reward
        elif distance >= self.config.max_goal_distance.m:
            # agent exceeded max distance from goal
            value = self.config.distance_reward
        elif in_docking and max_velocity_exceeded:
            # agent exceeded velocity constraint within docking region
            value = self.config.crash_reward
        elif in_docking and not max_velocity_exceeded:
            # agent safely made it to the docking region
            value = self.config.success_reward
            if self.config.timeout:
                # Add time reward component, if timeout specified
                value += 1 - (sim_time / self.config.timeout)

        return value
