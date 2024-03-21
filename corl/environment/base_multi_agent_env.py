"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import copy
import typing
from collections import OrderedDict

import gymnasium.utils.seeding
from pydantic import BaseModel
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from corl.agents.base_agent import BaseAgent
from corl.episode_parameter_providers import EpisodeParameterProvider
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.units import UnitRegistryConfiguration, corl_set_ureg
from corl.simulators.base_simulator import BaseSimulator, BaseSimulatorState


class BaseCorlMultiAgentEnvValidator(BaseModel):
    """Validation model for the inputs of BaseCorlMultiAgentEnv"""

    disable_action_history: bool = False  # disables the saving of actions during episode
    unit_reg_config: UnitRegistryConfiguration = UnitRegistryConfiguration()


class BaseCorlMultiAgentEnv(MultiAgentEnv):
    """Abstract base CoRL environment"""

    episode_parameter_provider_name: str = "environment"

    def __init__(self, config: dict):
        """Initialize base env"""
        env_config = copy.deepcopy(config)
        try:
            env_config_vars = vars(env_config)
        except TypeError:
            env_config_vars = {}

        ureg_validator = BaseCorlMultiAgentEnvValidator(unit_reg_config=env_config.get("unit_reg_config", UnitRegistryConfiguration()))
        ureg = ureg_validator.unit_reg_config.create_registry_from_args()
        corl_set_ureg(ureg)

        self.config: BaseCorlMultiAgentEnvValidator = self.get_validator()(**env_config, **env_config_vars)

        self._info: typing.OrderedDict = OrderedDict()
        self._done_info: typing.OrderedDict = OrderedDict()
        self._reward_info: typing.OrderedDict = OrderedDict()
        self._actions: list[typing.Any] = []
        self._observation: typing.OrderedDict = OrderedDict()
        self._agent_dict: dict[str, BaseAgent] = {}
        self._state: BaseSimulatorState
        self._simulator: BaseSimulator
        self._episode_id: int | None = None

    @staticmethod
    def get_validator() -> type[BaseCorlMultiAgentEnvValidator]:
        """Get the validator for this class."""
        return BaseCorlMultiAgentEnvValidator

    def post_process_trajectory(self, agent_id, batch, episode, policy):
        """easy accessor for calling post process trajectory
            correctly

        Arguments:
            agent_id: agent id
            batch: post processed Batch - be careful modifying
        """
        raise NotImplementedError

    @property
    def info(self) -> OrderedDict:
        """[summary]

        Returns:
            Union[OrderedDict, None] -- [description]
        """
        return self._info

    @property
    def done_info(self) -> OrderedDict:
        """[summary]

        Returns
        -------
        Union[OrderedDict, None]
            [description]
        """
        return self._done_info

    @property
    def reward_info(self) -> OrderedDict:
        """[summary]

        Returns
        -------
        Union[OrderedDict, None]
            [description]
        """
        return self._reward_info

    @property
    def state(self) -> BaseSimulatorState:
        """
        state of platform object.  Current state.

        Returns
        -------
        BaseSimulatorState
            the dict storing the current state of environment.
        """
        return self._state

    @property
    def simulator(self) -> BaseSimulator:
        """
        simulator simulator instance

        Returns
        -------
        BaseSimulator
            The simulator instance in the base
        """
        return self._simulator

    @property
    def observation(self) -> OrderedDict:
        """
        observation get the observation for the agents in this environment

        Returns
        -------
        OrderedDict
            the dict holding the observations for the agents
        """
        return self._observation

    @property
    def actions(self) -> list[typing.Any]:
        """
        get the current episode parameter provider episode id

        Returns
        -------
        list
            actions
        """
        return self._actions

    @property
    def episode_id(self) -> int | None:
        """
        get the current episode parameter provider episode id

        Returns
        -------
        int or None
            the episode id
        """
        return self.episode_id

    @property
    def local_variable_store(self) -> dict[str, typing.Any]:
        """
        get the local variable store

        Returns
        -------
        Dict
            the local variable store episode id
        """
        raise NotImplementedError

    @property
    def agent_dict(self) -> dict[str, BaseAgent]:
        """
        get the agent dict

        Returns
        -------
        Dict
            agents
        """
        return self._agent_dict

    @property
    def epp(self) -> EpisodeParameterProvider:
        """
        get the Episode Parameter Provider

        Returns
        -------
        EpisodeParameterProvider
        """
        raise NotImplementedError

    @property
    def glue_info(self) -> OrderedDict:
        """[summary]

        Returns:
            Union[OrderedDict, None] -- [description]
        """
        return self._info

    @staticmethod
    def _sanity_check(space: gymnasium.Space, space_sample: EnvSpaceUtil.sample_type) -> None:
        """
        Sanity checks a space_sample against a space
        1. Check to ensure that the sample from the integration base
           Fall within the expected range of values.

        Note: space_sample and space expected to match up on
        Key level entries

        Parameters
        ----------
        space: gymnasium.spaces.Space
            the space to check the sample against
        space_sample: EnvSpaceUtil.sample_type
            the sample to check if it is actually in the bounds of the space

        Returns
        -------
        OrderedDict:
            the scaled_observations
        """
        if not space.contains(space_sample):
            EnvSpaceUtil.deep_sanity_check_space_sample(space, space_sample)
