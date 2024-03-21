"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from corl.dones.done_func_base import DoneStatusCodes
from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.metrics.generator import MetricGeneratorTerminalEventScope
from corl.evaluation.metrics.metric import Metric
from corl.evaluation.metrics.types.nonterminals.vector import Vector
from corl.evaluation.metrics.types.terminals.discrete import Discrete
from corl.evaluation.metrics.types.terminals.string import String


# temporarily setting line 37 to 'DockingDoneStatus'
class StatusCode(MetricGeneratorTerminalEventScope):
    """Generates DoneStatusCodes value for the agent in an event

    Metric: Discrete
    Scope: Event
    """

    done_condition: str

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:
        if "agent_id" not in kwargs:
            raise RuntimeError('Expecting "agent_id" to be provided')

        agent_id = kwargs["agent_id"].split(".")[0]

        return Discrete[DoneStatusCodes](params.episode_state[agent_id][self.done_condition])


class DonesVec(MetricGeneratorTerminalEventScope):
    """Generates Vector of strings indicating the done condition

    Metric: Vector[String]
    Scope: Event
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:  # noqa: PLR6301
        if "agent_id" not in kwargs:
            raise RuntimeError('Expecting "agent_id" to be provided')

        agent_id = kwargs["agent_id"]

        platform_id_list = params.agent_to_platforms[agent_id]

        done_set = set()

        for platform_id in platform_id_list:
            dones = params.dones[platform_id]
            triggered_dones = [key for key in dones if dones[key] is True]
            done_set.update(triggered_dones)

        return Vector([String(item) for item in done_set])
