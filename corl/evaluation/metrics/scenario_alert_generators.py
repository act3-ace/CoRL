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

import typing

from pydantic import BaseModel

from .alerts import AlertGenerator


class ScenarioAlertGenerators(BaseModel):
    """Set of metric generators for a scenario
    """

    world: typing.List[AlertGenerator]
    default_participant: typing.List[AlertGenerator]
    specific_participant: typing.Dict[str, typing.List[AlertGenerator]]
    raise_on_error: bool = True

    @staticmethod
    def from_dict(data: dict, raise_on_error: bool = True) -> ScenarioAlertGenerators:
        """Create an instance from a dictionary
        """

        if "world" in data:
            world_alert_generator = []
            for alert in data["world"]:
                world_alert_generator.append(AlertGenerator(**alert, ))

        default_agent_alert_generators: typing.List[AlertGenerator] = []
        agent_alert_generators: typing.Dict[str, typing.List[AlertGenerator]] = {}
        if "agent" in data:
            for key in data["agent"]:
                if key == "__default__":
                    for alert in data["agent"]["__default__"]:
                        default_agent_alert_generators.append(AlertGenerator(**alert))
                else:
                    for alert in data["agent"][key]:
                        agent_alert_generators[key].append(AlertGenerator(**alert))

        return ScenarioAlertGenerators(
            world=world_alert_generator,
            default_participant=default_agent_alert_generators,
            specific_participant=agent_alert_generators,
            raise_on_error=raise_on_error,
        )
