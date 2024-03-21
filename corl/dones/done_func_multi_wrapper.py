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

from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator
from corl.simulators.base_simulator import BaseSimulatorState


class BaseMultiWrapperDoneValidator(DoneFuncBaseValidator):
    """
    wrapped - the wrapped done instance
    """

    wrapped: list[DoneFuncBase]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseMultiWrapperDone(DoneFuncBase):
    """A base object that dones can inherit in order to "wrap" a single done instance"""

    def __init__(self, **kwargs) -> None:
        self.config: BaseMultiWrapperDoneValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[BaseMultiWrapperDoneValidator]:
        return BaseMultiWrapperDoneValidator

    def dones(self) -> list[DoneFuncBase]:
        """Get the wrapped done instance"""
        return self.config.wrapped

    @abc.abstractmethod
    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: BaseSimulatorState,
        observation_space: gymnasium.Space,
        observation_units: OrderedDict,
    ) -> bool:
        ...
