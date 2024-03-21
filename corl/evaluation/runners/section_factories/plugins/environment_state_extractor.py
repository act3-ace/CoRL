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


class EnvironmentStateExtractor(ABC):
    """Abstract class to define how to extract information that needs to be saved from the larger environment state"""

    @abstractmethod
    def extract(self, state: dict) -> dict:
        """extract information from the given state"""
        raise RuntimeError("Must Instantiate this method")
