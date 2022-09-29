"""
This module defines the controller used by the agent to interact with its environment.
"""

import numpy as np

from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_parts import BaseController
from corl.simulators.docking_1d.available_platforms import Docking1dAvailablePlatformTypes
from corl.simulators.docking_1d.properties import ThrustProp
from corl.simulators.docking_1d.simulator import Docking1dSimulator


class Thrust1dController(BaseController):
    """
    A controller to apply thrust along a single axis.

    Parameters
    ----------
    parent_platform : Docking1dPlatform
        the platform to which the controller belongs
    config : dict
        contains configuration proprties
    control_properties : corl.libraries.property.BoxProp
        a class to define the acceptable bounds and units of the controller's control
    """

    def __init__(
        self,
        parent_platform,
        config,
        control_properties=ThrustProp,
    ):  # pylint: disable=W0102
        self.config: Thrust1dControllerValidator  # noqa # pylint: disable=undefined-variable
        super().__init__(property_class=control_properties, parent_platform=parent_platform, config=config)

    @property
    def name(self):
        """
        Returns
        -------
        String
            name of the controller
        """
        return self.config.name

    def apply_control(self, control: np.ndarray) -> None:
        """
        Applies control to the parent platform

        Parameters
        ----------
        control
            ndarray describing the control to the platform
        """
        self.parent_platform.save_action_to_platform(action=control)

    def get_applied_control(self) -> np.ndarray:
        """
        Retreive the applied control to the parent platform

        Returns
        -------
        np.ndarray
            Previously applied action

        """
        return self.parent_platform.get_applied_action()


# The defined controller must be registered with the PluginLibrary, along with a name for reference in the config, and
# a dict defining associated Simulator class and platform type enum.

PluginLibrary.AddClassToGroup(
    Thrust1dController,
    "1D_Controller_Thrust", {
        "simulator": Docking1dSimulator, "platform_type": Docking1dAvailablePlatformTypes.DOCKING1D
    }
)
