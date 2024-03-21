"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Logs extra environment metrics specific to Pong.
"""
from ray.rllib import BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2


class CustomPongMetrics(DefaultCallbacks):
    """Logs the number of ball hits for each paddle"""

    def on_episode_end(self, *, base_env: BaseEnv, episode: EpisodeV2, **_) -> None:  # noqa: PLR6301
        """At the end of each episode, log the total paddle hits"""
        if isinstance(episode, Exception):
            return

        env = base_env.get_sub_environments()[episode.env_id]
        # Cycle through the platforms and log the ball hit for each
        for platform_name, simulator in env.state.sim_platforms.items():
            key = f"{platform_name}/ball_hits"
            episode.custom_metrics[key] = simulator.paddle.ball_hits

            paddle_type = simulator.paddle_type.name
            key = f"{paddle_type}/ball_hits"
            episode.custom_metrics[key] = simulator.paddle.ball_hits
