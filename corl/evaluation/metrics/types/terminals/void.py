"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import typing

from corl.evaluation.metrics.metric import TerminalMetric


class Void(TerminalMetric):
    """Terminal Metric to represent a real number"""

    def __add__(self, rhs: typing.Any):
        raise RuntimeError(f"No logic on how to handle {type(rhs)}")

    def __truediv__(self, rhs: typing.Any):
        raise RuntimeError(f"No logic on how to handle {type(rhs)}")
