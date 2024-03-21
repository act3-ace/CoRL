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

from pydantic import BaseModel

from corl.evaluation.metrics.generator import MetricGenerator
from corl.libraries.factory import Factory
from corl.libraries.functor import Functor


class ScenarioMetricGenerators(BaseModel):
    """Represents a set of metric generators for a scenario"""

    world: list[MetricGenerator]
    default_participant: list[MetricGenerator]
    specific_participant: dict[str, list[MetricGenerator]]

    @staticmethod
    def _instantiate_config(config):
        """Iterate through keys for a config for a Functor and build anything that itself has a factory"""

        # Iteratively go though config and see if there is anything to use Factory for
        for config_key in config:
            if config[config_key] is not None and "type" in config[config_key]:
                config[config_key] = Factory.resolve_factory(config[config_key], {})

        return config

    @staticmethod
    def from_dict(data: dict) -> ScenarioMetricGenerators:
        """Create an instance from a dictionary"""

        if "world" in data:
            world_metric_generator = []
            for metric in data["world"]:
                functor = Functor(**metric)
                world_metric_generator.append(functor.create_functor_object())

        default_agent_metric_generators: list[MetricGenerator] = []
        agent_metric_generators: dict[str, list[MetricGenerator]] = {}
        if "agent" in data:
            for key in data["agent"]:
                if key == "__default__":
                    for metric in data["agent"]["__default__"]:
                        # Iteratively go though config and see if there is anything to use Factory for
                        functor = Functor(**metric)
                        default_agent_metric_generators.append(functor.create_functor_object())
                else:
                    for metric in data["agent"][key]:
                        functor = Functor(**metric)
                        if key not in agent_metric_generators:
                            agent_metric_generators[key] = []
                        agent_metric_generators[key].append(functor.create_functor_object())

        return ScenarioMetricGenerators(
            world=world_metric_generator,
            default_participant=default_agent_metric_generators,
            specific_participant=agent_metric_generators,
        )
