"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.metrics.generator import MetricGeneratorTerminalEventScope
from corl.evaluation.metrics.metric import Metric
from corl.evaluation.metrics.types.nonterminals.dict import Dict
from corl.evaluation.metrics.types.nonterminals.vector import Vector
from corl.evaluation.metrics.types.terminals.real import Real
from corl.evaluation.metrics.types.terminals.void import Void


class TotalReward(MetricGeneratorTerminalEventScope):
    """Generates single Real indicating the total reward for an event

    Metric: Real
    Scope: Event
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:  # noqa: PLR6301
        if "agent_id" not in kwargs:
            raise RuntimeError('Expecting "agent_id" to be provided')

        agent_id = kwargs["agent_id"].split(".")[0]

        # Find the last step with observation data from the platform in question
        # If the platform isn't in the episode at all, return Void
        steps_with_platform_in_question = [item for item in params.steps if agent_id in item.agents and item.agents[agent_id] is not None]
        if not steps_with_platform_in_question:
            return Void()

        last_step_with_platform_data = steps_with_platform_in_question[-1]

        if last_step_with_platform_data.agents[agent_id] is None:
            raise RuntimeError("Non Op")

        return Real(last_step_with_platform_data.agents[agent_id].total_reward)


class RewardVector(MetricGeneratorTerminalEventScope):
    """Generates vector of Dicts for the rewards calculated for each step during an event for an agent

    Metric: Vector[Dict]
    Scope: Event
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:  # noqa: PLR6301
        if "agent_id" not in kwargs:
            raise RuntimeError('Expecting "agent_id" to be provided')

        agent_id = kwargs["agent_id"].split(".")[0]

        arr: list[Metric] = []
        for step in params.steps:
            if agent_id not in step.agents or step.agents[agent_id] is None:
                break

            map_rewards = step.agents[agent_id].rewards

            if map_rewards is None:
                continue
            # Create a non terminal metric (Dict) that is comprised of the terminal (Real) rewards
            real_dict: dict[str, Metric] = {key: Real(map_rewards[key]) for key in map_rewards}
            arr.append(Dict(real_dict))

        return Vector(arr)
