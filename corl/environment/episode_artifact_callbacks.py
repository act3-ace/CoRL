"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Extra callbacks that can be for evaluating during training to
generate episode artifacts.
"""
from corl.evaluation import serialize_platforms
from corl.evaluation.runners.section_factories.engine.rllib.episode_artifact_logging_callback import EpisodeArtifactLoggingCallback


class PongEpisodeArtifactLogger(EpisodeArtifactLoggingCallback):
    """Logs episode artifacts for each evaluation epoch during training"""

    def platform_serializer(self):  # noqa: PLR6301
        return serialize_platforms.Serialize_Pong()


class Docking1dEpisodeArtifactLogger(EpisodeArtifactLoggingCallback):
    """Logs episode artifacts for each evaluation epoch during training"""

    def platform_serializer(self):  # noqa: PLR6301
        return serialize_platforms.serialize_Docking_1d()
