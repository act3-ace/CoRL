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
from corl.libraries.units import Quantity


@dataclasses.dataclass
class Real(TerminalMetric):
    """Terminal Metric to represent a real number"""

    value: float

    def __add__(self, rhs: typing.Any):
        if isinstance(rhs, Real):
            return Real(self.value + rhs.value)
        raise RuntimeError(f"No logic on how to handle {type(rhs)}")

    def __truediv__(self, rhs: typing.Any):
        lhs = self.value.value if isinstance(self.value, Real) else self.value

        if isinstance(rhs, (Real)):
            rhs = rhs.value
        elif isinstance(rhs, Quantity):
            rhs = rhs.m

        quotient = float(lhs) / float(rhs)

        return Real(quotient)

    def __lt__(self, rhs: typing.Any):
        if isinstance(rhs, Real):
            return self.value < rhs.value
        if isinstance(rhs, int):
            return self.value < typing.cast(float, rhs)
        if isinstance(rhs, float):
            return self.value < rhs
        raise RuntimeError(f"No logic on how to handle {type(rhs)}")

    def __gt__(self, rhs: typing.Any):
        if isinstance(rhs, Real):
            return self.value > rhs.value
        if isinstance(rhs, int):
            return self.value > typing.cast(float, rhs)
        if isinstance(rhs, float):
            return self.value > rhs
        raise RuntimeError(f"No logic on how to handle {type(rhs)}")
