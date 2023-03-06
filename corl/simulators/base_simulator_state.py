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
from collections import deque

from pydantic import BaseModel


class BaseSimulatorState(BaseModel):
    """
    Pydantic model containing a BaseClass for the state a simulator should return
    Other simulators may subclass this to return custom information
    """
    sim_platforms: typing.Dict
    sim_time: float
    user_data: typing.Dict = {}
    episode_history: typing.Optional[deque]
    episode_state: typing.Dict = {}
    agent_episode_state: typing.Dict = {}
    sim_update_rate: int = 1
