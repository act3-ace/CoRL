"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Reward for GymnasiumSimulator
"""
from corl.rewards.reward_func_base import RewardFuncBase


class GymnasiumReward(RewardFuncBase):
    """
    Reward for GymnasiumSimulator that rewards the reward
    coming from the simulator provided state and reports it
    """

    def __call__(
        self,
        observation,
        action,
        next_observation,
        state,
        next_state,
        observation_space,
        observation_units,
    ):
        return next_state.rewards[self.config.platform_names[0]]
