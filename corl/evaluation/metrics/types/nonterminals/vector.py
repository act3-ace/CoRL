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

from corl.evaluation.metrics.metric import Metric, NonTerminalMetric
from corl.evaluation.metrics.types.nonterminals.timed_value import TimedValue


class Vector(NonTerminalMetric):
    """NonTerminal Metric to contain a vector of other metrics

    This class can be inherited extend other vectorized data, like TimeSeries
    """

    _arr: list[Metric]
    _is_time_series: bool

    def __init__(self, arr: list[Metric]):
        self._arr = arr

        # Not sure what a "generator" is
        self._is_time_series = all(isinstance(item, TimedValue) for item in self._arr)

    @property
    def arr(self) -> list[Metric]:
        """Get list of metrics stored by the vector"""
        return self._arr

    def is_time_series(self) -> bool:
        """Determine if the vector is a time series"""
        return self._is_time_series

    def __add__(self, rhs: typing.Any):
        raise NotImplementedError()

    def __truediv__(self, rhs: typing.Any):
        raise NotImplementedError()
