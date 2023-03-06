"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
property definitions for sensors
"""
import typing

import numpy as np
from pydantic import Field, StrictFloat, StrictStr
from typing_extensions import Annotated

from corl.libraries.property import BoxProp


class TimeProp(BoxProp):
    """
    TimeProp defines time space
    """
    name: str = "time"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.inf]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["seconds"]
    description: str = "The time of the sensor"


class NoOpProp(BoxProp):
    """
    TimeProp defines time space
    """
    name: str = "NoOp"
    low: Annotated[typing.List[StrictFloat], Field(min_items=0, max_items=0)] = []
    high: Annotated[typing.List[StrictFloat], Field(min_items=0, max_items=0)] = []
    unit: Annotated[typing.List[StrictStr], Field(min_items=0, max_items=0)] = []
    description: str = "No op prop actions from this pace will not be sent to simulator"
