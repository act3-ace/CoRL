"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Configures teams to perform in evaluation
"""
import dataclasses
import typing

import jsonargparse

import corl.evaluation.loader.i_agent_loader as agent_loader


@jsonargparse.typing.final
@dataclasses.dataclass
class Agent:
    """An Agent on a platform
    """
    name: str
    config: str
    platforms: typing.List[str]
    policy: str
    agent_loader: agent_loader.IAgentLoader


@jsonargparse.typing.final
@dataclasses.dataclass
class Platform:
    """A platform that participates in a team
    """
    name: str
    config: str


@jsonargparse.typing.final
@dataclasses.dataclass
class Teams:
    """Describes teams that are to participate in an evaluation
    """
    platform_config: typing.List[Platform]
    agent_config: typing.List[Agent]

    def iterate_on_participant(self, func: typing.Callable[[Agent], typing.Any]) -> typing.List[typing.Any]:
        """Perform some action on each particpant and aggregate results to an array

        Arguments:
            func {typing.Callable[Agent]} -- function to perform

        Returns:
            [type] -- Aggregated array from each particpant on each team.
        """
        arr = []
        for agent in self.agent_config:
            arr.append(func(agent))

        return arr
