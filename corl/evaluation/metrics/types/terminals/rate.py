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

import dataclasses
import typing

from corl.evaluation.metrics.metric import TerminalMetric


@dataclasses.dataclass
class Rate(TerminalMetric):
    """Terminal Metric to represent the number of occurrences of an event vs total"""

    occurrence: int
    total: int

    def __add__(self, rhs: typing.Any) -> Rate:
        if isinstance(rhs, Rate):
            return Rate(self.occurrence + rhs.occurrence, self.total + rhs.total)
        raise NotImplementedError()

    def __truediv__(self, denominator: typing.Any):
        raise RuntimeError("Can not divide a Rate metric")

    def __lt__(self, rhs: typing.Any) -> bool:
        if isinstance(rhs, Rate):
            raise NotImplementedError()
        if isinstance(rhs, int):
            return self.occurrence / self.total < rhs / self.total
        if isinstance(rhs, float):
            return self.occurrence / self.total < float(rhs)
        raise RuntimeError(f"Unknown how to compute lt operator between Rate and {type(rhs)}")

    def __gt__(self, rhs: typing.Any) -> bool:
        if isinstance(rhs, Rate):
            raise NotImplementedError()
        if isinstance(rhs, int):
            return self.occurrence / self.total > rhs / self.total
        if isinstance(rhs, float):
            return self.occurrence / self.total > rhs
        raise RuntimeError(f"Unknown how to compute lt operator between Rate and {type(rhs)}")

    def __sub__(self, rhs: typing.Any) -> Rate:
        if isinstance(rhs, Rate):
            raise NotImplementedError()
        if isinstance(rhs, int):
            return Rate(self.occurrence - rhs, self.total)
        if isinstance(rhs, float):
            return Rate(int((self.occurrence - rhs) * self.total), self.total)
        raise RuntimeError(f"Unknown how to subtract between Rate and {type(rhs)}")
