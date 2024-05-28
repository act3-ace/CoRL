# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
"""
Extra callbacks that can be for evaluating during training to
generate episode artifacts.
"""
import typing
from abc import abstractmethod

from gymnasium import Env

from corl.environment.multi_agent_env import ACT3MultiAgentEnv


class CorlGymEnv(Env):
    """gymnaisum Env that wraps the ACT3MultiAgentEnv so it can be used with other training frameworks"""

    def __init__(self, config: dict[str, typing.Any]) -> None:
        self._corl_ma_env = ACT3MultiAgentEnv(config)

        # Convert the spaces to gymnasium spaces
        self.action_space = self._convert_action_space()
        self.observation_space = self._convert_obs_space()

        self._episode_length = 0

    def step(self, action):

        if self._episode_length == 0:
            action = self.action_space.sample()

        corl_actions = self._convert_action(action)

        trainable_observations, trainable_rewards, trainable_dones, dones, trainable_info = self._corl_ma_env.step(corl_actions)

        return (
            self._convert_obs(trainable_observations),
            next(iter(trainable_rewards.values())),
            any(trainable_dones.values()),
            any(dones.values()),
            trainable_info,
        )

    def reset(self, *, seed=None, options=None):
        # Get the trainable items from the base environment
        self._episode_length = 0
        training_obs, info = self._corl_ma_env.reset(seed=seed, options=options)
        return self._convert_obs(training_obs), info

    @abstractmethod
    def _convert_action_space(self):
        """
        Override with desired action space conversion
        """

        raise NotImplementedError

    def _convert_action(self, action):
        """
        Override with desired action conversion
        """
        raise NotImplementedError

    def _convert_obs_space(self):
        """
        Override with desired observation space conversion
        """
        raise NotImplementedError

    def _convert_obs(self, obs):
        """
        Override with desired observation conversion
        """
        raise NotImplementedError
