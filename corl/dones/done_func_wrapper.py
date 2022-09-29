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

from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator
from corl.libraries.environment_dict import DoneDict, StateDict


class BaseWrapperDoneValidator(DoneFuncBaseValidator):
    """
    wrapped - the wrapped done instance
    """
    wrapped: DoneFuncBase

    class Config:  # pylint: disable=C0115, R0903
        arbitrary_types_allowed = True


class BaseWrapperDone(DoneFuncBase):
    """A base object that dones can inherit in order to "wrap" a single done instance
    """

    def __init__(self, **kwargs) -> None:
        self.config: BaseWrapperDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[BaseWrapperDoneValidator]:
        return BaseWrapperDoneValidator

    def done(self) -> DoneFuncBase:
        """Get the wrapped done instance
        """
        return self.config.wrapped

    @abc.abstractmethod
    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> DoneDict:
        ...
