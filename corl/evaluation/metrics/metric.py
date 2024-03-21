"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import abc
import typing


class Metric:
    """Abstract class that any metric is to inherit from"""

    @abc.abstractmethod
    def __add__(self, rhs: typing.Any):
        ...

    @abc.abstractmethod
    def __truediv__(self, rhs: typing.Any):
        ...


class TerminalMetric(abc.ABC, Metric):  # Allows for enhanced type checking
    """Abstract class that a terminal metric is to inherit from.

    A Terminal metric is one that is not contains pure data.
    Examples: Real, String, etc
    """


class NonTerminalMetric(abc.ABC, Metric):  # Allows for enhanced type checking
    """Abstract class that a nonterminal metric is to inherit from.

    A NonTerminal metric is a metric that itself contains other metrics.
    Example: Vector of Reals
    """
