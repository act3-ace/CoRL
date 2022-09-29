# pylint: disable=no-self-argument, no-self-use
"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import collections
import copy
import fractions
import typing
from itertools import chain
from typing import Any, Dict, List, MutableMapping, Union

import flatten_dict
import gym
from pydantic import BaseModel, PyObject, validator

from corl.dones.done_func_base import DoneFuncBase
from corl.dones.episode_length_done import EpisodeLengthDone
from corl.episode_parameter_providers import EpisodeParameterProvider, Randomness
from corl.glues.base_glue import BaseAgentGlue
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.environment_dict import DoneDict, RewardDict
from corl.libraries.factory import Factory
from corl.libraries.functor import Functor, FunctorDictWrapper, FunctorMultiWrapper, FunctorWrapper, ObjectStoreElem
from corl.libraries.parameters import Parameter
from corl.libraries.plugin_library import PluginLibrary
from corl.rewards.reward_func_base import RewardFuncBase
from corl.simulators.base_simulator import validation_helper_units_and_parameters


class PartsFunctor(BaseModel):
    """
    - part: The name of the part. This should be registered to a corresponding platform part class in the plugin library.
    - config: The specific configuration dictionary expected by the platform part.
    """
    part: str
    config: Dict[str, Any] = {}
    references: Dict[str, str] = {}


class AgentParseBase(BaseModel):
    """
    - agent: The BaseAgent class representing the agent.
    - config: The agent specific configuration including glues, reward, dones, parts, etc.
    """
    agent: PyObject  # this autoimports (no more plugin loader for glues)
    config: Dict[str, Any]

    @validator('agent')
    def check_agent(cls, v):
        """Check if agent subclass AgentBase"""
        if not issubclass(v, BaseAgent):
            raise ValueError(f"Agents must subclass BaseAgent, but is of type {v}")
        return v


class AgentParseInfo(BaseModel):
    """
    - class_config: The agent class and its configuration
    - platform_name: The name of the platform
    - policy_config: Configuration of the policy of this agent
    """
    class_config: AgentParseBase
    platform_name: str
    policy_config: Dict[str, Any]


class PlatformParseInfo(BaseModel):
    """
    - platform_config
    """
    platform_config: Dict[str, Any]


class BaseAgentEppParameters(BaseModel):
    """
    glues: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        keys: glue name, parameter name

    rewards: typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]] = {}
        keys: reward name, ? name, parameter name

    dones: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        keys: done name, parameter name

    reference_store: typing.Dict[str, typing.Any] = {}
        keys: reference name

    simulator_reset: typing.Dict[str, typing.Any] = {}
        keys: whatever the simulator wants, but it needs to be kwargs to simulator reset
    """
    glues: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
    rewards: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
    dones: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
    reference_store: typing.Dict[str, typing.Any] = {}
    simulator_reset: typing.Dict[str, typing.Any] = {}

    @staticmethod
    def _validate_leaves_are_parameters(obj):
        if isinstance(obj, dict):
            for _key, value in obj.items():
                BaseAgentEppParameters._validate_leaves_are_parameters(value)
        elif not isinstance(obj, Parameter):
            raise TypeError(f"Invalid type: {type(obj)} (required type: {Parameter.__qualname__})")

    @validator('glues', 'rewards', 'dones', 'reference_store', 'simulator_reset')
    def validate_leaves_are_parameters(cls, v):
        """
        Verifies the outer most leaf nodes of a config are parameter types
        """
        BaseAgentEppParameters._validate_leaves_are_parameters(v)
        return v


class BaseAgentParser(BaseModel):
    """
    - parts: A list of part configurations.
    - glues: A list of glue functor configurations.
    - rewards: A optional list of reward functor configurations
    - dones: An optional list of done functor configurations
    """
    # these can be sensors, controller, or other platform parts
    parts: List[PartsFunctor]  # this uses plugin loader to load
    reference_store: Dict[str, ObjectStoreElem] = {}
    glues: List[Union[FunctorMultiWrapper, FunctorWrapper, Functor, FunctorDictWrapper]]
    rewards: List[Union[FunctorMultiWrapper, FunctorWrapper, Functor, FunctorDictWrapper]] = []
    dones: List[Union[FunctorMultiWrapper, FunctorWrapper, Functor, FunctorDictWrapper]] = []
    simulator_reset_parameters: Dict[str, Any] = {}
    # frame_rate in Hz (i.e. .25 indicates this agent will process when sim_tim % 4 == 0)
    frame_rate: float = 1.0
    agent_name: str
    platform_name: str
    multiple_workers: bool = False

    episode_parameter_provider: Factory
    episode_parameter_provider_parameters: BaseAgentEppParameters = None  # type: ignore
    epp: EpisodeParameterProvider = None  # type: ignore

    # this is a dictionary validated by the simulator that allows the agent to tell
    # the simulator what to do with this agent

    class Config:
        """Allow arbitrary types for Parameter"""
        arbitrary_types_allowed = True

    @validator('glues', each_item=True)
    def check_glues(cls, v):
        """Check if glues subclass BaseAgentGlue"""
        if not issubclass(v.functor, BaseAgentGlue):
            raise TypeError(f"Glue functors must subclass BaseAgentGlue, but glue {v.name} is of type {v.functor}")
        return v

    @validator('rewards', each_item=True)
    def check_rewards(cls, v):
        """Check if rewards subclass RewardFuncBase"""
        if not issubclass(v.functor, RewardFuncBase):
            raise TypeError(f"Reward functors must subclass RewardFuncBase, but reward {v.name} is of type {v.functor}")
        return v

    @validator('dones', each_item=True)
    def check_dones(cls, v):
        """Check if dones subclass DoneFuncBase"""
        if not issubclass(v.functor, DoneFuncBase):
            raise TypeError(f"Done functors must subclass DoneFuncBase, but done {v.name} is of type {v.functor}")
        if issubclass(v.functor, EpisodeLengthDone):
            raise ValueError("Cannot specify EpisodeLengthDone as it is automatically added")
        return v

    resolve_factory = validator('reference_store', pre=True, each_item=True, allow_reuse=True)(Factory.resolve_factory)

    @validator('simulator_reset_parameters', pre=True)
    def update_units_and_parameters(cls, v):
        """Update simulation reset parameters to meet base simulator requirements."""
        return validation_helper_units_and_parameters(v)

    @validator('episode_parameter_provider')
    def check_epp(cls, epp_factory, values):
        """Check if episode parameter provider subclass EpisodeParameterProvider"""
        if not issubclass(epp_factory.type, EpisodeParameterProvider):
            raise TypeError(f"Episode parameter providers must subclass EpisodeParameterProvider, but is is of type {epp_factory.type}")

        # replace %%AGENT%% with agent_name
        epp_params = flatten_dict.flatten(epp_factory.config)
        for key, value in epp_params.items():
            if isinstance(value, str):
                epp_params[key] = value.replace(r'%%AGENT%%', values['agent_name'])
        epp_factory.config = flatten_dict.unflatten(epp_params)

        return epp_factory

    @validator('episode_parameter_provider_parameters', always=True, pre=True)
    def build_episode_parameter_provider_parameters(cls, _v, values) -> BaseAgentEppParameters:
        """Create the episode parameter provider config"""

        for key in ['reference_store', 'glues', 'rewards', 'dones', 'simulator_reset_parameters']:
            assert key in values

        reference_parameters: typing.Dict[str, Parameter] = {}
        for ref_name, ref_value in values['reference_store'].items():
            if isinstance(ref_value, Parameter):
                reference_parameters[ref_name] = ref_value

        glue_parameters: typing.Dict[str, typing.Dict[str, Parameter]] = {}
        for functor in values['glues']:
            functor.add_to_parameter_store(glue_parameters)

        reward_parameters: typing.Dict[str, typing.Dict[str, Parameter]] = {}
        for functor in values['rewards']:
            functor.add_to_parameter_store(reward_parameters)

        done_parameters: typing.Dict[str, typing.Dict[str, Parameter]] = {}
        for functor in values['dones']:
            functor.add_to_parameter_store(done_parameters)

        flat_data = flatten_dict.flatten(values['simulator_reset_parameters'])
        for key, value in flat_data.items():
            if isinstance(value, BaseModel):
                flat_data[key] = value.dict()
        expanded_sim_reset_params = flatten_dict.unflatten(flat_data)

        sim_parameters_flat = {
            name: param
            for name, param in flatten_dict.flatten(expanded_sim_reset_params).items()
            if isinstance(param, Parameter)
        }
        sim_parameters = flatten_dict.unflatten(sim_parameters_flat)

        return BaseAgentEppParameters(
            glues=glue_parameters,
            rewards=reward_parameters,
            dones=done_parameters,
            reference_store=reference_parameters,
            simulator_reset=sim_parameters
        )

    @validator('epp', always=True, pre=True)
    def build_epp(cls, epp, values):
        """Builds an instance of an EpisodeParameterProvider if necessary"""
        if epp is None:
            assert 'episode_parameter_provider_parameters' in values

            epp_params = dict(values['episode_parameter_provider_parameters'])
            flat_epp_params = flatten_dict.flatten(epp_params)
            epp = values['episode_parameter_provider'].build(parameters=flat_epp_params)
        return epp

    @validator('frame_rate', always=True)
    def simplify_frame_rate(cls, v):
        """Expand the precision of the frame rate (e.g. .333 -> .3333333333334)"""
        f = fractions.Fraction(v).limit_denominator(20)
        return f.numerator / f.denominator


class BaseAgent:  # pylint: disable=too-many-public-methods
    """
    Base class representing an agent in an environment.
    """

    def __init__(self, **kwargs) -> None:
        self.config: BaseAgentParser = self.get_validator(**kwargs)
        self.agent_glue_dict: typing.Dict[str, BaseAgentGlue] = {}
        self.agent_reward_dict = RewardDict()
        self.agent_done_dict = DoneDict()
        # self._agent_glue_obs_export_behavior = {}

        # Sample parameter provider
        # This RNG only used here.  Normal use uses the one from the environment.
        rng, _ = gym.utils.seeding.np_random(0)
        self.fill_parameters(rng=rng, default_parameters=True)

    @property
    def platform_name(self) -> str:
        """
        Returns the name of the platform this agent is attached to

        Returns
        -------
        str:
            The name of the platform
        """
        return self.config.platform_name

    @property
    def trainable(self) -> bool:
        """
        Flag denoting if agent is trainable.

        Returns
        -------
        bool:
            False
        """
        return False

    @property
    def get_validator(self) -> typing.Type[BaseAgentParser]:
        """
        Get the validator used to validate the kwargs passed to BaseAgent.

        Returns
        -------
        BaseAgentParser
            A BaseAgent kwargs parser and validator.
        """
        return BaseAgentParser

    @property
    def frame_rate(self) -> float:
        """Get the frame rate this agent runs at"""
        return self.config.frame_rate

    def fill_parameters(self, rng: Randomness, default_parameters: bool = False) -> None:
        """Sample the episode parameter provider to fill the local variable store."""
        if default_parameters:
            current_parameters = self.config.epp.config.parameters
        else:
            current_parameters, _ = self.config.epp.get_params(rng)
        self.local_variable_store = flatten_dict.unflatten({k: v.get_value(rng) for k, v in current_parameters.items()})

    def get_simulator_reset_parameters(self) -> typing.Dict[str, typing.Any]:
        """Return the local parameters needed within the simulator reset"""
        output = copy.deepcopy(self.config.simulator_reset_parameters)

        def expand_and_update(data: MutableMapping[str, Any], simulator_reset: typing.Mapping[str, Any]):
            for key, value in data.items():
                if isinstance(value, BaseModel):
                    expand_and_update(vars(value), simulator_reset.get(key, {}))
                elif isinstance(value, collections.abc.MutableMapping):
                    expand_and_update(value, simulator_reset.get(key, {}))
                else:
                    if key in simulator_reset:
                        data[key] = simulator_reset[key]

        expand_and_update(output, self.local_variable_store.get('simulator_reset', {}))

        return output

    def get_platform_parts(self, simulator, platform_type):
        """
        Gets a list of the agent's platform parts.

        Parameters
        ----------
        simulator: Simulator class used by environment to simulate agent actions.
        platform_type: Platform type enumeration corresponding to the agent's platform

        Returns
        -------
        list:
            List of platform parts
        """
        my_parts = []
        for part in self.config.parts:
            tmp = PluginLibrary.FindMatch(part.part, {"simulator": simulator, "platform_type": platform_type})
            for ref_name, ref_key in part.references.items():
                if ref_key not in self.config.reference_store:
                    raise RuntimeError(f'Part reference {ref_key} must be in the agent reference store.')
                ref_value = self.config.reference_store[ref_key]
                if isinstance(ref_value, Parameter):
                    raise TypeError(f'Part reference {ref_key} cannot be Parameter')
                part.config[ref_name] = ref_value
            my_parts.append((tmp, part.config))
        return my_parts

    def make_glues(self, platform, agent_id: str, env_ref_stores: typing.List[typing.Dict[str, typing.Any]]) -> None:
        """
        Creates agent glue functors from agent configuration.

        Parameters
        ----------
        platform: The platform instance associated with the glue functors.
        agent_id: The id of the agent associated with the glue functors.
        env_ref_stores: Reference stores for items managed by the environment

        Returns
        -------
        None
        """
        self.agent_glue_dict.clear()
        for glue_dict in self.config.glues:
            created_glue = glue_dict.create_functor_object(
                platform=platform,
                agent_name=agent_id,
                param_sources=[self.local_variable_store.get('glues', {})],
                ref_sources=[self.local_variable_store.get('reference_store', {}), self.config.reference_store] + env_ref_stores
            )
            glue_name = created_glue.get_unique_name()
            if glue_name is not None:
                if glue_name in self.agent_glue_dict:
                    raise RuntimeError(f"The {glue_name} glue has a unique name, but it already exists")
            else:
                raise RuntimeError(f"No glue name for {created_glue}")

            # add the glue to the agent glue dict
            # self._agent_glue_obs_export_behavior[glue_name] = glue.training_obs_behavior
            self.agent_glue_dict[glue_name] = created_glue

    def make_rewards(self, agent_id: str, env_ref_stores: typing.List[typing.Dict[str, typing.Any]]) -> None:
        """
        Creates agent reward functors from agent configuration.

        Parameters
        ----------
        agent_id: The id of the agent associated with the reward functors.
        env_ref_stores: Reference stores for items managed by the environment

        Returns
        -------
        None
        """
        tmp = []
        for reward_dict in self.config.rewards:
            tmp.append(
                reward_dict.create_functor_object(
                    agent_name=agent_id,
                    param_sources=[self.local_variable_store.get('rewards', {})],
                    ref_sources=[self.local_variable_store.get('reference_store', {}), self.config.reference_store] + env_ref_stores
                )
            )
        self.agent_reward_dict = RewardDict(processing_funcs=tmp)

    def make_dones(
        self,
        agent_id: str,
        platform_name: str,
        dones: typing.Iterable[Functor],
        env_params: typing.List[typing.Sequence[typing.Dict[str, typing.Any]]],
        env_ref_stores: typing.List[typing.Dict[str, typing.Any]]
    ) -> None:
        """
        Creates agent done functors from agent configuration.

        Parameters
        ----------
        agent_id: The id of the agent associated with the done functors.
        dones: Additional done conditions to apply
        env_params: Parameters for the provided dones
        env_ref_stores: Reference stores for items managed by the environment

        Returns
        -------
        None
        """

        tmp = []
        for done_dict in chain(self.config.dones, dones):
            tmp.append(
                done_dict.create_functor_object(
                    param_sources=[self.local_variable_store.get('dones', {})] + env_params,
                    ref_sources=[self.local_variable_store.get('reference_store', {}), self.config.reference_store] + env_ref_stores,
                    agent_name=agent_id,
                    platform_name=platform_name
                )
            )
        self.agent_done_dict = DoneDict(processing_funcs=tmp)

    def create_space(self, space_getter: typing.Optional[typing.Callable] = None):
        """
        Creates a gym dict space from the agent's glues.

        Parameters
        ----------
        space_getter (optional): A function that takes a glue_obj: BaseAgentGlue and returns a space.
            Default is space_getter=lambda glue_obj: glue_obj.observation_space().

        Returns
        -------
        gym.spaces.dict.Dict()
            A gym dict space composed of all the spaces returned by applying the space_getter to the agent glues.
        """
        if space_getter is None:

            def default_getter(glue_obj):
                return glue_obj.observation_space()

            space_getter = default_getter

        return_space = gym.spaces.dict.Dict()
        # loop over all glue name and  glue_obj pairs
        glue_obj: BaseAgentGlue
        for glue_name, glue_obj in self.agent_glue_dict.items():
            # call our space getter to pick which space we want,
            # for example: action_space, observation_space, normalized_action_space, normalized_observation_space
            space_def = space_getter(glue_obj)
            # if the space is None don't add it
            if space_def:
                return_space.spaces[glue_name] = space_def
        return return_space if len(return_space.spaces) > 0 else None

    def apply_action(self, action_dict: typing.Dict[typing.Any, typing.Any]):
        """
        Applies actions to agent.

        Parameters
        ----------
        action_dict (optional): A dictionary of actions to be applied to agent.

        Returns
        -------
        None
        """
        raw_action_dict = collections.OrderedDict()
        obs = self.get_observations()
        for glue_name, glue_object in self.agent_glue_dict.items():
            if glue_name in action_dict:
                normalized_action = action_dict[glue_name]
                raw_action = glue_object.unnormalize_action(normalized_action)
                raw_action_dict[glue_name] = raw_action
                glue_object.apply_action(raw_action, obs)
        return raw_action_dict

    def create_next_action(
        self,
        observation: typing.Dict,  # pylint: disable=unused-argument
        reward: float,  # pylint: disable=unused-argument
        done: bool,  # pylint: disable=unused-argument
        info: typing.Dict  # pylint: disable=unused-argument
    ):
        """
        Creates next action agent will apply.

        Parameters
        ----------
        action_dict (optional): A dictionary of actions to be applied to agent.

        Returns
        -------
        None
        """
        ...

    def get_observations(self):
        """
        Gets combined observation from agent glues.

        Returns
        -------
        OrderedDict
            A dictionary of glue observations in the form {glue_name: glue_observation}
        """
        return_observation: collections.OrderedDict = collections.OrderedDict()
        for glue_name, glue_object in self.agent_glue_dict.items():
            glue_obs = glue_object.get_observation()
            if glue_obs:
                return_observation[glue_name] = glue_obs
        return return_observation

    def get_info_dict(self):
        """
        Gets combined observation from agent glues.

        Returns
        -------
        OrderedDict
            A dictionary of glue observations in the form {glue_name: glue_observation}
        """
        return_info: collections.OrderedDict = collections.OrderedDict()
        for glue_name, glue_object in self.agent_glue_dict.items():
            glue_info = glue_object.get_info_dict()
            if glue_info:
                return_info[glue_name] = glue_info
        return return_info

    def get_dones(self, observation, action, next_observation, next_state, observation_space, observation_units):
        """
        Get agent's done state from agent done functors.

        Parameters
        ----------
        - observation: The observation dictionary used to compute action.
        - action: The action computed from observation.
        - next_observation: The observation dictionary containing observation of next_state.
        - next_state: The state dictionary containing the environment state after action was applied to the environment.
        - observation_space: The agent observation space.
        - observation_units: The units of the observations in the observation space. This may be None.

        Returns
        -------
        DoneDict[str: bool]
            A dictionary of the agent's done state in the form {agent_id: done}
        """
        return self.agent_done_dict(
            observation=observation,
            action=action,
            next_observation=next_observation,
            next_state=next_state,
            observation_space=observation_space,
            observation_units=observation_units,
        )

    def get_rewards(self, observation, action, next_observation, state, next_state, observation_space, observation_units):
        """
        Get agent's environment rewards from agent reward functors.

        Parameters
        ----------
        - observation: The observation dictionary used to compute action.
        - action: The action computed from observation.
        - next_observation: The observation dictionary containing observation of next_state.
        - state: The state dictionary containing the environment state before action was applied to the environment.
        - next_state: The state dictionary containing the environment state after action was applied to the environment.
        - observation_space: The agent observation space.
        - observation_units: The units of the observations in the observation space. This may be None.

        Returns
        -------
        RewardDict[str: Any]
            A dictionary of the agent's reward in the form {agent_id: reward}
        """
        return self.agent_reward_dict(
            observation=observation,
            action=action,
            next_observation=next_observation,
            state=state,
            next_state=next_state,
            observation_space=observation_space,
            observation_units=observation_units
        )

    def post_process_trajectory(self, agent_id, state, batch, episode, policy, reward_info):
        """
        calling function for sending data to on_postprocess_trajectory

        Arguments:
            agent_id str -- name of the agent
            state {[type]} -- current simulator state
            batch {[type]} -- batch from the current trajectory
            episode {[type]} -- the episode object
            policy {[type]} -- the policy that ran the trajectory
            reward_info {[type]} -- the info dict for rewards
        """
        for reward_glue in self.agent_reward_dict.process_callbacks:
            output_value = reward_glue.post_process_trajectory(agent_id, state, batch, episode, policy)
            if output_value is not None:

                # If we have a bad setup the agent may be "dead" on first step
                # This in turn causes major issues when trying to index reward_info
                # Check that the agent exists
                if agent_id in reward_info:
                    reward_info[agent_id].setdefault(reward_glue.name, {}).setdefault(agent_id, 0.0)
                    reward_info[agent_id][reward_glue.name][agent_id] += output_value

    def get_glue(self, name: str) -> typing.Optional[BaseAgentGlue]:
        """
        Get the glue object with the given name
        """
        if name in self.agent_glue_dict:
            return self.agent_glue_dict[name]
        return None

    def normalize_observation(self, obs_name: str, obs: EnvSpaceUtil.sample_type) -> typing.Optional[EnvSpaceUtil.sample_type]:
        """
        Normalize a single observation value (if a corresponding glue is found)

        Parameters
        ----------
        - obs_name: The name of the observation
        - obs: the observation (value)

        Returns
        -------
        Normalized observation value, or None (if no corresponding glue is found)
        """
        glue_obj = self.get_glue(obs_name)
        if glue_obj is not None:
            # TODO: fix this
            if 'ObserveSensorRepeated' in obs_name:
                return glue_obj.normalize_observation(copy.deepcopy(obs))
            return glue_obj.normalize_observation(obs)
        return None

    def normalize_observations(self, observations: collections.OrderedDict) -> collections.OrderedDict:
        """
        Normalizes glue observations according to glue definition.

        Parameters
        ----------
        - observations: A dictionary of observation glues.

        Returns
        -------
        OrderedDict[str: typing.Union[np.ndarray, typing.Tuple, typing.Dict]]
            A dictionary of scaled observations in the form {glue_name: scaled_observation}
        """
        normalized_observation_dict = collections.OrderedDict()
        for obs_name, obs in observations:
            normalized_obs = self.normalize_observation(obs_name, obs)
            if normalized_obs:
                normalized_observation_dict[obs_name] = normalized_obs

        return normalized_observation_dict

    def set_removed(self, removed_state: bool):
        """
        Set the agent_removed flag for the agent's glues.

        Parameters
        ----------
        - removed_state (bool): The value to set the agent_removed flag to.

        Returns
        -------
        None
        """
        for glue in self.agent_glue_dict.values():
            glue.set_agent_removed(removed_state)


class TrainableBaseAgent(BaseAgent):
    """
    Base class representing a trainable agent in an environment.
    """

    @property
    def trainable(self) -> bool:
        """
        Flag denoting if agent is trainable.

        Returns
        -------
        bool:
            True
        """
        return True
