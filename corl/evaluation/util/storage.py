"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Module contains a storage interface which can be extended to implement various kinds of storage for the evaluation framework.
"""

from abc import ABC, abstractmethod


class Storage(ABC):
    """
    Abstract class that defines an interface by which various kinds of storage can be utilized
    """

    def __init__(self) -> None:
        ...

    @abstractmethod
    def load_artifacts_location(self, config: dict[str, str]):
        """
        Load information about the locations of various artifacts
        """

    @abstractmethod
    def store(self):
        """
        Logic to perform the storage of data.
        """
