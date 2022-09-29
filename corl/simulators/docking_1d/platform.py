"""
This module extends corl.simulators.base_platform.BasePlatform to create a simple one dimensional
docking platform.
"""

import typing

import numpy as np

from corl.simulators.base_platform import BasePlatform, BasePlatformValidator
from corl.simulators.docking_1d.entities import Deputy1D


class Docking1dPlatformValidator(BasePlatformValidator):
    """Docking1dPlatformValidator

    Parameters
    ----------
        platform: Deputy1D
            Deputy associated with the CoRL Docking1dPlatform
    """
    platform: Deputy1D


class Docking1dPlatform(BasePlatform):
    """
    A platform representing a spacecraft operating under Double Integrator dynamics.
    Allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function

    Parameters
    ----------
    platform_name : str
        Name of the platform
    platform : sim_entity
        Backend simulation entity associated with the platform
    platform_config : dict
        Platform-specific configuration dictionary
    """

    def __init__(self, **kwargs) -> None:
        self.config: Docking1dPlatformValidator
        super().__init__(**kwargs)

        self._platform = self.config.platform
        self._last_applied_action = np.array([0], dtype=np.float32)  # thrust
        self._sim_time = 0.0

    @property
    def get_validator(self) -> typing.Type[Docking1dPlatformValidator]:
        return Docking1dPlatformValidator

    def __eq__(self, other):
        if isinstance(other, Docking1dPlatform):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and self.sim_time == other.sim_time
            return eq
        return False

    def get_applied_action(self):
        """
        returns the action stored in this platform

        Returns:
            typing.Any -- any sort of stored action
        """
        return self._last_applied_action

    def save_action_to_platform(self, action):
        """
        saves an action to the platform if it matches
        the action space

        Arguments:
            action typing.Any -- The action to store in the platform
        """
        if isinstance(action, np.ndarray) and len(action) == 1:
            self._last_applied_action = action

    @property
    def position(self):
        """
        The position of the platform

        Returns
        -------
        np.ndarray
            The position vector of the platform
        """
        return self._platform.position

    @property
    def velocity(self):
        """
        The velocity of the platform

        Returns
        -------
        np.ndarray
            The velocity vector of the platform
        """
        return self._platform.velocity

    @property
    def sim_time(self):
        """
        The current simulation time in seconds.

        Returns
        -------
        float
            Current simulation time
        """
        return self._sim_time

    @sim_time.setter
    def sim_time(self, time):
        self._sim_time = time

    @property
    def operable(self):
        return True
