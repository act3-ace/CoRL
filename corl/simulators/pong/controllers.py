"""
This module defines the controller used by the agent to interact with its environment.
"""

import pygame

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import DiscreteProp
from corl.simulators.base_parts import BaseController
from corl.simulators.pong.available_platforms import PongAvailablePlatformTypes
from corl.simulators.pong.paddle_platform import PaddlePlatform, PaddleType
from corl.simulators.pong.simulator import PongSimulator


class PaddleMoveProp(DiscreteProp):
    """
    Paddle movement control property

    Parameters:
    name : str
        control property name
    n : int
        number of discrete controls
    unit : str
        unit of measurement for control input
    description : str
        description of control properties
    """

    name: str = "paddle_move"
    n: int = 3
    description: str = "Paddle Move [UP, NO_MOVE, DOWN]"


class PaddleController(BaseController):
    """
    A controller to change the position of a paddle.

    Parameters
    ----------
    parent_platform : PaddlePlatform
        the platform to which the controller belongs
    config : dict
        contains configuration properties
    control_properties : corl.libraries.property.DiscreteProp
        a class to define the acceptable bounds and units of the controller's control
    """

    def __init__(
        self,
        parent_platform: PaddlePlatform,
        config,
        control_properties=PaddleMoveProp,
    ):  # pylint: disable=W0102
        super().__init__(property_class=control_properties, parent_platform=parent_platform, config=config)
        self._last_applied_action = 0

    @property
    def name(self):
        """
        Returns
        -------
        String
            name of the controller
        """
        return self.config.name

    def apply_control(self, control: int) -> None:  # type: ignore
        """
        Applies control to the parent platform

        Parameters
        ----------
        control
            ndarray describing the control to the platform
        """
        self._last_applied_action = control
        self.parent_platform.last_move = self.convert_action(control)

    def get_applied_control(self) -> int:  # type: ignore
        """
        Retreive the applied control to the parent platform

        Returns
        -------
        int
            Previously applied action

        """
        return self._last_applied_action

    def convert_action(self, discrete_action) -> int:
        """
        Retreive the applied control to the parent platform

        Returns
        -------
        np.ndarray
            Previously applied action

        """
        if discrete_action == 1:
            return pygame.K_0  # any non action key

        # Left Paddle
        if self.parent_platform.paddle_type == PaddleType.LEFT:
            if discrete_action == 0:
                return pygame.K_w
            return pygame.K_s

        # Right Paddle
        if discrete_action == 0:
            return pygame.K_UP
        return pygame.K_DOWN


# The defined controller must be registered with the PluginLibrary, along with a name for reference in the config, and
# a dict defining associated Simulator class and platform type enum.

PluginLibrary.AddClassToGroup(
    PaddleController, "PaddleController", {
        "simulator": PongSimulator, "platform_type": PongAvailablePlatformTypes.PADDLE
    }
)
