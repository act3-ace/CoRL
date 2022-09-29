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
import typing
from collections import OrderedDict

from pydantic import BaseModel

from corl.libraries.env_func_base import EnvFuncBase
from corl.libraries.environment_dict import RewardDict
from corl.libraries.state_dict import StateDict


class RewardFuncBaseValidator(BaseModel):
    """
    name: Name of reward functor
    agent_name: Name of agent the reward functor belongs to
    """
    name: typing.Optional[str]
    agent_name: str


class RewardFuncBase(EnvFuncBase):
    """The base implementation for reward functors
    """

    def __init__(self, **kwargs):
        self.config: RewardFuncBaseValidator = self.get_validator(**kwargs)

    @property
    def get_validator(self) -> typing.Type[RewardFuncBaseValidator]:
        """Returns pydantic validator associated with this class
        """
        return RewardFuncBaseValidator

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

    def post_process_trajectory(self, agent_id, state, batch, episode, policy):  # pylint: disable=unused-argument,no-self-use
        """Allows the user to modify the trajectory of the episode
        in the batch collected during an rllib callback. WARNING: This function is dangerous
        you can completly destroy training using this
        Use it only as a last resort
        """
        ...

    @property
    def name(self) -> str:
        """ gets the name fo the functor

        Returns
        -------
        str
            The name of the functor
        """
        return type(self).__name__ if self.config.name is None else self.config.name
