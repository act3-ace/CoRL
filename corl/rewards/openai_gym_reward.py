"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Reward for OpenAIGymSimulator
"""
from corl.libraries.environment_dict import RewardDict
from corl.rewards.reward_func_base import RewardFuncBase


class OpenAIGymReward(RewardFuncBase):
    """
    Reward for OpenAiGymSimulator that rewards the reward
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
        reward_dict = RewardDict()
        reward_dict[self.config.agent_name] = next_state.rewards[self.config.platform_names[0]]

        return reward_dict
