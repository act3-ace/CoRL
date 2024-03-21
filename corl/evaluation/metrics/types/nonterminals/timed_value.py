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

from pydantic import BaseModel, ConfigDict, field_validator

from corl.evaluation.metrics.metric import Metric, NonTerminalMetric
from corl.libraries.units import Quantity, corl_get_ureg


class TimedValue(BaseModel, NonTerminalMetric):
    """NonTerminal Metric that contains a metric which occurred at a specific time"""

    time: Quantity
    value: Metric
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("time")
    @classmethod
    def time_unit_must_be_time(cls, v: Quantity):
        """Enusure that the units given are of Time Dimension"""
        assert v.u == corl_get_ureg().get_unit("second")
        return v

    def __add__(self, rhs: typing.Any):
        raise NotImplementedError()

    def __truediv__(self, rhs: typing.Any):
        raise NotImplementedError()
