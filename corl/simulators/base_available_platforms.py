"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
import enum
from typing import Any, Dict


class BaseAvailablePlatformTypes(enum.Enum):
    """Enumeration that outlines the platform types that have been implemented
    """

    @abc.abstractclassmethod
    def ParseFromNameModel(cls, config: Dict[str, Any]) -> BaseAvailablePlatformTypes:  # pylint: disable=unused-argument
        """
        ParseFromNameModel is responsible for returning the platform type being used
        by a platform configuration that will be provided by some platform configuration

        Arguments:
            config {Dict[str, Any]} -- The platform configuration for this platform

        Returns:
            BaseAvailablePlatformTypes -- The platform type being used by this platform
        """
        ...
