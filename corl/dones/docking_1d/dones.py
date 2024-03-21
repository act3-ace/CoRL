"""
This module defines functions that determine terminal conditions for the 1D Docking environment.
"""

import numpy as np

from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from corl.libraries.units import Quantity
from corl.simulators.common_platform_utils import get_platform_by_name, get_sensor_by_name


class DockingDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains the docking region and crash velocity data needed for
    computations in the SuccessfulDockingDoneFunction.
    """

    docking_region_radius: Quantity
    velocity_threshold: Quantity
    position_sensor_name: str
    velocity_sensor_name: str


class DockingDoneFunction(DoneFuncBase):
    """
    A done function that determines if deputy has successfully docked with the chief or not.
    """

    REQUIRED_UNITS = {"docking_region_radius": "meter", "velocity_threshold": "meter / second"}

    def __init__(self, **kwargs) -> None:
        self.config: DockingDoneValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[DockingDoneValidator]:
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        SuccessfulDockingDoneValidator
            config validator for the SuccessfulDockingDoneFunction

        """
        return DockingDoneValidator

    def __call__(self, observation, action, next_observation, next_state, observation_space, observation_units):
        """
        Parameters
        ----------
        observation : np.ndarray
            np.ndarray describing the current observation
        action : np.ndarray
            np.ndarray describing the current action
        next_observation : np.ndarray
            np.ndarray describing the incoming observation
        next_state : np.ndarray
            np.ndarray describing the incoming state

        Returns
        -------
        done : DoneDict
            dictionary containing the condition condition for the current agent

        """
        deputy = get_platform_by_name(next_state, self.config.platform_name)

        position_sensor = get_sensor_by_name(deputy, self.config.position_sensor_name)
        velocity_sensor = get_sensor_by_name(deputy, self.config.velocity_sensor_name)

        position = position_sensor.get_measurement()
        velocity = velocity_sensor.get_measurement()

        chief_position = np.array([0])
        docking_region_radius = self.config.docking_region_radius

        distance = abs(position.m - chief_position).item()
        in_docking = distance <= docking_region_radius.m

        max_velocity_exceeded = self.config.velocity_threshold.m < abs(velocity.m).item()

        successful_dock = in_docking and not max_velocity_exceeded
        crash = in_docking and max_velocity_exceeded

        done = False
        if crash:
            done = True
            next_state.episode_state[self.platform][self.name] = DoneStatusCodes.LOSE
        elif successful_dock:
            done = True
            next_state.episode_state[self.platform][self.name] = DoneStatusCodes.WIN

        return done
