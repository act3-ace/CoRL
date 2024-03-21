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
import typing
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict

import corl.evaluation.loader.i_agent_loader as agent_loader
from corl.evaluation.evaluation_factory import FactoryLoader
from corl.parsers.agent_and_platform import CorlAgentConfigArgs, CorlPlatformConfigArgs


class LoadableCorlAgent(CorlAgentConfigArgs):
    """An Agent on a platform"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_loader: Annotated[agent_loader.IAgentLoader, BeforeValidator(FactoryLoader.resolve_factory)]


class Teams(BaseModel):
    """Describes teams that are to participate in an evaluation"""

    platform_config: list[CorlPlatformConfigArgs]
    agent_config: list[LoadableCorlAgent]

    def iterate_on_participant(self, func: typing.Callable[[LoadableCorlAgent], typing.Any]) -> list[typing.Any]:
        """Perform some action on each particpant and aggregate results to an array

        Arguments:
            func {typing.Callable[Agent]} -- function to perform

        Returns:
            [type] -- Aggregated array from each particpant on each team.
        """
        return [func(agent) for agent in self.agent_config]
