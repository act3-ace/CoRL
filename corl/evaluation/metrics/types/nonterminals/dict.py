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

from corl.evaluation.metrics.metric import Metric, NonTerminalMetric


@dataclasses.dataclass
class Dict(NonTerminalMetric):  # noqa: PLW1641
    """NonTerminal Metric to contain a dictionary of other metrics"""

    data: dict[str, Metric]

    def __add__(self, rhs: typing.Any):
        if isinstance(rhs, Dict):
            if self.data.keys() != rhs.data.keys():
                raise RuntimeError("Keys do not match when adding two Dict metrics")

            return Dict({key: self.data[key] + rhs.data[key] for key in self.data})
        raise RuntimeError(f"Can not add {type(rhs)} to a Dict metric")

    def __eq__(self, rhs: typing.Any):
        raise NotImplementedError()

    def __truediv__(self, denominator: typing.Any):
        if isinstance(denominator, int):
            return Dict({key: self.data[key] / denominator for key in self.data})
        raise RuntimeError(f"Can not divide a Dict metric by  {type(denominator)}")
