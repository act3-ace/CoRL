"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Callbacks to do logging of episodes
"""
from ray.rllib import BaseEnv, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


class EpisodeInterrupted(Exception):
    def __init__(self):
        super().__init__("EpisodeInterrupted")


class InterruptableCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: dict[str, callable] | None = None):  # type: ignore
        super().__init__(legacy_callbacks_dict)  # type: ignore

        self._reset_required = False

    # def on_episode_start(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     base_env: BaseEnv,
    #     policies: dict[PolicyID, Policy],
    #     episode: Episode | EpisodeV2,
    #     env_index: int | None = None,
    #     **kwargs,
    # ) -> None:
    #     self._reset_required = False

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: dict[PolicyID, Policy] | None = None,
        episode: Episode | EpisodeV2,
        env_index: int | None = None,
        **kwargs,
    ) -> None:
        if self._reset_required:
            self._reset_required = False
            raise EpisodeInterrupted

    def stop(self):
        self._reset_required = True
