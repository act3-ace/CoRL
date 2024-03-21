"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import enum


class Scopes(enum.IntEnum):
    """Scopes which a metric can be computed on"""

    EVENT = enum.auto()
    """An Event is a single flight
    """

    EVALUATION = enum.auto()
    """An Evaluation is a list of events
    """

    TOURNAMENT = enum.auto()
    """A Tournament is a list of evaluations
    """


def from_string(name: str) -> Scopes:
    """Generate a scope enumeration from string"""
    if name == "event":
        return Scopes.EVENT
    if name == "evaluation":
        return Scopes.EVALUATION
    if name == "tournament":
        return Scopes.TOURNAMENT
    raise RuntimeError(f"Unknown metric scope: {name}")
