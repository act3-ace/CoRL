"""
This module defines an enumeration of available platform types that can be referenced in the platform config.
See config/tasks/docking_1d/docking1d_platform.yml.
"""

from __future__ import annotations

from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_platform_type import BasePlatformType
from corl.simulators.docking_1d.simulator import Docking1dSimulator


class Docking1dAvailablePlatformType(BasePlatformType):
    """_summary_

    Args:
        PlatformType (_type_): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """

    @classmethod
    def match_model(cls, config: dict) -> bool:
        if config["name"] == "DOCKING1D":
            return True
        return False


PluginLibrary.add_platform_to_sim(Docking1dAvailablePlatformType, Docking1dSimulator)
