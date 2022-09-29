"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Provides AvailablePlatformTypes for the Open AI Gym Simulator
"""
from __future__ import annotations

from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_available_platforms import BaseAvailablePlatformTypes
from corl.simulators.openai_gym.gym_simulator import OpenAIGymSimulator


class OpenAIGymAvailablePlatformTypes(BaseAvailablePlatformTypes):
    """Enumeration that outlines the platform types that have been implemented
    """
    MAIN = (1, )

    @classmethod
    def ParseFromNameModel(cls, config: dict) -> OpenAIGymAvailablePlatformTypes:
        """
        This just returns the main platform type, as openai gym simulators are simple
        """
        return OpenAIGymAvailablePlatformTypes.MAIN


PluginLibrary.AddClassToGroup(OpenAIGymAvailablePlatformTypes, "OpenAIGymSimulator_Platforms", {"simulator": OpenAIGymSimulator})
