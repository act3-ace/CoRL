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
import typing
from collections import OrderedDict

from corl.libraries.environment_dict import RewardDict
from corl.libraries.state_dict import StateDict
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator


class BaseWrapperRewardValidator(RewardFuncBaseValidator):
    """
    wrapped - the wrapped reward instance
    """
    wrapped: RewardFuncBase

    class Config:  # pylint: disable=C0115, R0903
        arbitrary_types_allowed = True


class BaseWrapperReward(RewardFuncBase):
    """A base object that rewards can inherit in order to "wrap" a single reward instance
    """

    def __init__(self, **kwargs) -> None:
        self.config: BaseWrapperRewardValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[BaseWrapperRewardValidator]:
        return BaseWrapperRewardValidator

    def reward(self) -> RewardFuncBase:
        """Get the wrapped reward instance
        """
        return self.config.wrapped

    @abc.abstractmethod
    def __call__(
        self,
        observation: OrderedDict,
        action,
        next_observation: OrderedDict,
        state: StateDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> RewardDict:
        ...
