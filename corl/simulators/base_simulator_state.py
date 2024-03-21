"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""


from pydantic import BaseModel, Extra


class BaseSimulatorState(BaseModel, extra=Extra.forbid):
    """
    Pydantic model containing a BaseClass for the state a simulator should return
    Other simulators may subclass this to return custom information
    """

    sim_platforms: dict
    sim_time: float
    user_data: dict = {}
    # episode_history: defaultdict[str, deque] = defaultdict(partial(deque, maxlen=1000))
    episode_state: dict = {}
    agent_episode_state: dict = {}
    simulator_info: dict = {}
    sim_update_rate_hz: float = 1
