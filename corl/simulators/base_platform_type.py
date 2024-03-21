"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from abc import abstractmethod


class BasePlatformType:
    """
    Class describing a type of platform in a simulation
    This object will be used to register parts to the plugin library
    and help the simulator properly initialize platforms that may have
    custom methods of initialization
    """

    @classmethod
    @abstractmethod
    def match_model(cls, config: dict) -> bool:
        """
        this function takes in the platform configuration
        description for this platform and must return True
        if the config dict should be of this platform type
        or false otherwise
        """
