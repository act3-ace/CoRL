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
from typing import Annotated

import numpy as np
from pydantic import Field, StrictFloat, StrictStr

from corl.libraries.property import BoxProp, DiscreteProp


class TimeProp(BoxProp):
    """
    TimeProp defines time space
    """

    name: str = "time"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [np.inf]
    unit: StrictStr = "seconds"
    description: str = "The time of the sensor"


class NoOpProp(DiscreteProp):
    """
    NoOpProp defines null action
    """

    name: str = "NoOp"
    n: int = 1
    description: str = "No op prop actions from this space will not be sent to simulator"
