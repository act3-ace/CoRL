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
from enum import Enum

import gymnasium
from pydantic import BaseModel

from corl.libraries.env_func_base import EnvFuncBase
from corl.libraries.environment_dict import DoneDict
from corl.libraries.state_dict import StateDict
from corl.simulators.base_parts import BaseTimeSensor
from corl.simulators.base_simulator import BaseSimulatorState


class DoneStatusCodes(Enum):
    """reward states for the done conditions"""

    WIN = 1
    PARTIAL_WIN = 2
    DRAW = 3
    PARTIAL_LOSS = 4
    LOSE = 5


class DoneFuncBaseValidator(BaseModel):
    """Initialize the done condition

    All done conditions have three common parameters, with any others being handled by subclass validators

    Parameters
    ----------
    name : str
        A name applied to this done condition, by default the name of the class.
    agent_name : str
        Name of the agent to which this done condition applies, This is only valid
        if the done condition came from an agent
    platform_name : str
        Name of the platform to which this done condition applies
    """

    name: str = ""
    agent_name: str | None = None
    platform_name: str


class DoneFuncBase(EnvFuncBase):
    """Base implementation for done functors"""

    def __init__(self, **kwargs) -> None:
        self.config: DoneFuncBaseValidator = self.get_validator()(**kwargs)
        self.config.name = self.config.name if self.config.name else type(self).__name__

    @staticmethod
    def get_validator() -> type[DoneFuncBaseValidator]:
        """
        get validator for this Done Functor

        Returns:
            DoneFuncBaseValidator -- validator the done functor will use to generate a configuration
        """
        return DoneFuncBaseValidator

    @property
    def agent(self) -> str | None:
        """The agent to which this done is applied"""
        return self.config.agent_name

    @property
    def platform(self) -> str:
        """The platform to which this done is applied"""
        return self.config.platform_name

    @property
    def name(self) -> str:
        return self.config.name

    @staticmethod
    def _get_platform_time(platform):
        if sensor := [s for s in platform.sensors if isinstance(s, BaseTimeSensor)]:
            return sensor[0].get_measurement()[0]

        raise ValueError("Did not find time sensor type (BaseTimeSensor)")

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


class SharedDoneFuncBaseValidator(BaseModel):
    """
    name : str
        The name of this done condition
    """

    name: str = ""


class SharedDoneFuncBase(EnvFuncBase):
    """Base implementation for global done functors

    Global done functors are not associated with a single agent, but the environment as a whole.  They use a modified call syntax that
    receives the done dictionary and done info from the per-agent done functors.
    """

    ALL = "__all__"

    def __init__(self, **kwargs) -> None:
        self.config: SharedDoneFuncBaseValidator = self.get_validator()(**kwargs)
        self.config.name = self.config.name if self.config.name else type(self).__name__

    @staticmethod
    def get_validator() -> type[SharedDoneFuncBaseValidator]:
        """
        gets the validator for this SharedDoneFuncBase

        Returns:
            SharedDoneFuncBaseValidator -- validator for this class
        """
        return SharedDoneFuncBaseValidator

    @property
    def name(self) -> str:
        return self.config.name

    @abc.abstractmethod
    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
        local_dones: DoneDict,
        local_done_info: OrderedDict,
    ) -> DoneDict:
        ...
