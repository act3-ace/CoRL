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

from corl.dones.done_func_base import DoneStatusCodes


@dataclasses.dataclass
class EpisodeArtifact:
    """Artifact generated by an episode
    """

    @dataclasses.dataclass
    class AgentStep:
        """Data collected from perspective a an agent at each step
        """

        observations: dict
        actions: dict
        rewards: typing.Optional[typing.Dict[str, float]]
        total_reward: float

    @dataclasses.dataclass
    class Step:
        """Data collected at each timestep
        """

        agents: typing.Dict[str, EpisodeArtifact.AgentStep]
        """Dictionary of data for each agent
        """

        platforms: typing.List[dict]
        """Serialized platform data
        """

        environment_state: typing.Dict[str, typing.Any]
        """Dictionary of environment specific things
        The exact details of this field depends on the environment
        """

    @dataclasses.dataclass
    class DoneConfig:
        """Configurations that define dones"""

        task: typing.Dict[str, typing.List]
        world: typing.List[typing.Any]

    test_case: int
    worker_index: int
    params: typing.Dict[str, typing.Any]
    parameter_values: typing.Dict[str, typing.Any]  # full parameters set for episode (environment's + agent's local_variable_stores)
    artifacts_filenames: typing.Dict[str, str]
    wall_time_sec: float
    frame_rate: typing.Optional[float]
    steps: typing.List[EpisodeArtifact.Step]
    dones: typing.Dict[str, typing.Dict[str, bool]]
    episode_state: typing.Dict[str, typing.Dict[str, DoneStatusCodes]]
    observation_units: typing.Dict[str, typing.Any]
    platform_to_agents: typing.Dict[str, typing.Any]
    agent_to_platforms: typing.Dict[str, typing.Any]
    done_config: EpisodeArtifact.DoneConfig