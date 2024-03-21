"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import dataclasses
import typing

from corl.evaluation.metrics.metric import TerminalMetric


@dataclasses.dataclass
class String(TerminalMetric):  # noqa: PLW1641
    """Terminal Metric to represent a string"""

    value: str

    def __add__(self, rhs: typing.Any):
        raise RuntimeError("Can not add to a String metric")

    def __eq__(self, rhs: typing.Any):
        return self.value == rhs

    def __truediv__(self, denominator: typing.Any):
        raise RuntimeError("Can not divide a String metric")
