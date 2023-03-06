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

import jsonargparse
from pydantic.dataclasses import dataclass


@dataclass
@jsonargparse.typing.final
class CorlPlatformConfigArgs:
    """
    name: name of the platform
    config: path to the config file this platform should be configured with
    """
    name: str
    config: str


@dataclass
@jsonargparse.typing.final
class CorlPlatformsConfigArgs:
    """
    platform_config: list of platforms
    """
    platform_config: typing.List[CorlPlatformConfigArgs]


@dataclass
@jsonargparse.typing.final
class CorlAgentConfigArgs:
    """
    name: name of the agent
    platforms: a list of platforms the agent at least partially controls
    config: path to the agent config for this agent
    policy: path to the policy config for this agent
    """
    name: str
    platforms: typing.List[str]
    config: str
    policy: str


@dataclass
@jsonargparse.typing.final
class CorlAgentsConfigArgs:
    """
    agent_config: list of agents
    """
    agent_config: typing.List[CorlAgentConfigArgs]
