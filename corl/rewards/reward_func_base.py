"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Reward Functor Base Module
"""
import abc
from collections import OrderedDict

import gymnasium
from pydantic import BaseModel

from corl.libraries.env_func_base import EnvFuncBase
from corl.simulators.base_simulator_state import BaseSimulatorState


class RewardFuncBaseValidator(BaseModel):
    """
    name: Name of reward functor
    agent_name: Name of agent the reward functor belongs to
    platform_names: list of platforms controlled by the agent
                    this reward belongs to
    """

    name: str | None = None
    agent_name: str
    platform_names: list[str]


class RewardFuncBase(EnvFuncBase):
    """The base implementation for reward functors"""

    def __init__(self, **kwargs) -> None:
        self.config: RewardFuncBaseValidator = self.get_validator()(**kwargs)

    @staticmethod
    def get_validator() -> type[RewardFuncBaseValidator]:
        """Returns pydantic validator associated with this class"""
        return RewardFuncBaseValidator

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

    def post_process_trajectory(self, agent_id, state, batch, episode, policy):
        """Allows the user to modify the trajectory of the episode
        in the batch collected during an rllib callback. WARNING: This function is dangerous
        you can completely destroy training using this
        Use it only as a last resort
        """

    @property
    def name(self) -> str:
        """gets the name for the functor

        Returns
        -------
        str
            The name of the functor
        """
        return type(self).__name__ if self.config.name is None else self.config.name
