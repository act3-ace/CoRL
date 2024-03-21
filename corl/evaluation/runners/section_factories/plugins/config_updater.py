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


class ConfigUpdate(ABC):
    """Abstract class to update a configuration"""

    @abstractmethod
    def update(self, config: dict) -> dict:
        """Perform update

        Args:
            config (dict): config to update

        Returns:
            dict: updated configuration
        """
        raise RuntimeError("Must Instantiate this method")
