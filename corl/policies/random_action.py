"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Module with base implementations for Observations
"""
from collections import defaultdict

import flatten_dict
import numpy as np
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class RandomActionPolicy(Policy):
    """Random action policy."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.view_requirements = {key: value for key, value in self.view_requirements.items() if key != SampleBatch.PREV_ACTIONS}

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        if not isinstance(obs_batch, list):
            obs_batch = [obs_batch]

        actions_batch = defaultdict(list)
        for _ in obs_batch:
            random_actions = self.action_space.sample()
            for key, value in flatten_dict.flatten(random_actions).items():
                actions_batch[key].append(value)

        for key, value in actions_batch.items():
            actions_batch[key] = np.array(value)

        return flatten_dict.unflatten(actions_batch), [], {}

    def learn_on_batch(self, samples):  # noqa: PLR6301
        return {}

    def get_weights(self):  # noqa: PLR6301
        return {}

    def set_weights(self, weights):  # noqa: PLR6301
        return
