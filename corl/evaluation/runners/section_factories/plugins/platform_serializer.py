"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod

from corl.simulators.base_platform import BasePlatform


class PlatformSerializer(ABC):
    """Abstract class to define how to serialize a platform for saving"""

    @abstractmethod
    def serialize(self, platform: BasePlatform) -> dict:
        """Serialize provided platform

        Args:
            platform (BasePlatform): Platform to serialize

        Returns:
            dict: Platform serialized into a dict
        """
        raise RuntimeError("Must Instantiate this method")
