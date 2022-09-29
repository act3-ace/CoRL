"""
This module defines an enumeration of available platform types that can be referenced in the platform config.
See config/tasks/docking_1d/docking1d_platform.yml.
"""

from __future__ import annotations

from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_available_platforms import BaseAvailablePlatformTypes
from corl.simulators.docking_1d.simulator import Docking1dSimulator


class Docking1dAvailablePlatformTypes(BaseAvailablePlatformTypes):
    """
    An enumeration that outlines the platform types that have been implemented
    """

    DOCKING1D = (1, )

    @classmethod
    def ParseFromNameModel(cls, config: dict):
        """Given a config with the keys "model" and "name" determine the PlatformType

        Raises:
            RuntimeError: if the given config doesn't have both "name" and "model" keys
            RuntimeError: if the "name" and "model" keys do not match a known model
        """

        if "name" not in config:
            raise RuntimeError("Attempting to parse a PlatformType from name/model config, but both are not given!")

        if config["name"] == "DOCKING1D":
            return Docking1dAvailablePlatformTypes.DOCKING1D

        raise RuntimeError(f'name: {config["name"]} and model: {config["model"]} did not match a known platform type')


# BaseAvailablePlatformTypes children must be registed with the PluginLibrary, associated with a name, and their
# environment's Simulator class.

PluginLibrary.AddClassToGroup(Docking1dAvailablePlatformTypes, "Docking1dSimulator_Platforms", {"simulator": Docking1dSimulator})
