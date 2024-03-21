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
from corl.evaluation.metrics.types.terminals.real import Real


class EpisodeLength_Steps(MetricGeneratorTerminalEventScope):
    """Real number indicating seconds the event took

    Metric: Real
    Scope: Event
    """

    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:  # noqa: PLR6301
        return Real(len(params.steps))
