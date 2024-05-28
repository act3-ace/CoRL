# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
"""
This module defines an enumeration of available platform types that can be referenced in the platform config.
See config/tasks/docking_1d/docking1d_platform.yml.
"""

from __future__ import annotations

from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_platform_type import BasePlatformType
from corl.simulators.pong.simulator import PongSimulator


class PongAvailablePlatformType(BasePlatformType):
    """_summary_

    Args:
        PlatformType (_type_): _description_

    Returns:
        _type_: _description_
    """

    @classmethod
    def match_model(cls, config: dict) -> bool:
        """_summary_

        Args:
            config (dict): _description_

        Returns:
            bool: _description_
        """
        if config["name"] == "PADDLE":
            return True
        return False


# BaseAvailablePlatformTypes children must be registered with the PluginLibrary, associated with a name, and their
# environment's Simulator class.

PluginLibrary.add_platform_to_sim(PongAvailablePlatformType, PongSimulator)
