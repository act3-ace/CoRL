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

from pydantic import BaseModel, validator

from corl.evaluation.metrics.metric import Metric, NonTerminalMetric
from corl.libraries import units


class TimedValue(BaseModel, NonTerminalMetric):
    """NonTerminal Metric that contains a metric which occured at a specific time
    """

    time: units.ValueWithUnits
    value: Metric

    class Config:  # pylint:disable=too-few-public-methods # Needed for pydantic
        """Config for pydantic
        """
        arbitrary_types_allowed = True

    @validator('time')
    def time_unit_must_be_time(cls, v: units.ValueWithUnits):  # pylint: disable=no-self-argument
        """Enusure that the units given are of Time Demension
        """
        assert isinstance(v.units, units.Time)
        return v

    def __add__(self, rhs: typing.Any):
        raise NotImplementedError()

    def __truediv__(self, rhs: typing.Any):
        raise NotImplementedError()
