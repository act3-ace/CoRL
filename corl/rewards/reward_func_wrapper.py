"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import abc
from collections import OrderedDict

import gymnasium
from pydantic import ConfigDict

from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from corl.simulators.base_simulator_state import BaseSimulatorState


class BaseWrapperRewardValidator(RewardFuncBaseValidator):
    """
    wrapped - the wrapped reward instance
    """

    wrapped: RewardFuncBase
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseWrapperReward(RewardFuncBase):
    """A base object that rewards can inherit in order to "wrap" a single reward instance"""

    def __init__(self, **kwargs) -> None:
        self.config: BaseWrapperRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[BaseWrapperRewardValidator]:
        return BaseWrapperRewardValidator

    def reward(self) -> RewardFuncBase:
        """Get the wrapped reward instance"""
        return self.config.wrapped

    @abc.abstractmethod
    def __call__(
        self,
        observation: OrderedDict,
        action,
        next_observation: OrderedDict,
        state: BaseSimulatorState,
        next_state: BaseSimulatorState,
        observation_space: gymnasium.Space,
        observation_units: OrderedDict,
    ) -> float:
        ...
