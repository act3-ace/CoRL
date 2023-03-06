# pylint: disable=no-self-argument
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
from functools import lru_cache
from graphlib import TopologicalSorter
from typing import Any, Dict, List, Literal, MutableMapping, Union

import flatten_dict
import gym
from pydantic import BaseModel, PyObject, validator

from corl.dones.done_func_base import DoneFuncBase
from corl.dones.episode_length_done import EpisodeLengthDone
from corl.episode_parameter_providers import EpisodeParameterProvider, Randomness
from corl.glues.base_glue import BaseAgentGlue, TrainingExportBehavior
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.environment_dict import RewardDict
from corl.libraries.factory import Factory
from corl.libraries.functor import Functor, FunctorDictWrapper, FunctorMultiWrapper, FunctorWrapper, ObjectStoreElem
from corl.libraries.parameters import Parameter
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.space_transformations import SpaceTransformationBase
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
    platform_names: typing.List[str]
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


class AgentSpaceTransformation(BaseModel):
    """
    AgentSpaceTransformation

        transformation: Type[SpaceTransformationBase]
        config: Dict
    """
    transformation: PyObject
    config: typing.Dict = {}

    @validator("transformation")
    def validate_transformation(cls, val):
        """
        validates transformation subclass type
        """
        if issubclass(val, SpaceTransformationBase):
            return val

        raise TypeError(f"Invalid output_type {type(val)} in SpaceTypeConversionValidator")


class BaseAgentParser(BaseModel):
    """
    - parts: A list of part configurations.
    - glues: A list of glue functor configurations.
    - rewards: A optional list of reward functor configurations
    - dones: An optional list of done functor configurations
    """
    # these can be weapons, sensors, or controller
    agent_name: str
    parts: List[PartsFunctor]  # this uses plugin loader to load
    reference_store: Dict[str, ObjectStoreElem] = {}
    glues: List[Union[FunctorMultiWrapper, FunctorWrapper, Functor, FunctorDictWrapper]]
    rewards: List[Union[FunctorMultiWrapper, FunctorWrapper, Functor, FunctorDictWrapper]] = []
    dones: List[Union[FunctorMultiWrapper, FunctorWrapper, Functor, FunctorDictWrapper]] = []
    simulator_reset_parameters: Dict[str, Any] = {}
    # frame_rate in Hz (i.e. .25 indicates this agent will process when sim_tim % 4 == 0)
    frame_rate: float = 1.0
    platform_names: typing.List[str]
    multiple_workers: bool = False
    policy_action_transformation: typing.Optional[AgentSpaceTransformation] = None
    policy_observation_transformation: typing.Optional[AgentSpaceTransformation] = None
    # modifies the normalization for any glue created by this agent
    normalization_master_switch: Literal["ALL_ON", "DEFAULT", "ALL_OFF"] = "DEFAULT"
    episode_parameter_provider: Factory
    episode_parameter_provider_parameters: BaseAgentEppParameters = None  # type: ignore
    epp: EpisodeParameterProvider = None  # type: ignore

    # this is a dictionary validated by the simulator that allows the agent to tell
    # the simulator what to do with this agent

    class Config:
        """Allow arbitrary types for Parameter"""
        arbitrary_types_allowed = True

    @validator('glues', each_item=True)
    def check_glues(cls, v, values):
        """Check if glues subclass BaseAgentGlue"""
        if not issubclass(v.functor, BaseAgentGlue):
            raise TypeError(f"Glue functors must subclass BaseAgentGlue, but glue {v.name} is of type {v.functor}")

        assert 'agent_name' in values
        assert 'parts' in values

        cls._check_glue_parts(v, values)

        return v

    @classmethod
    def _check_glue_parts(cls, v, values) -> None:

        for part_type in ['sensor', 'controller']:
            if part_type in v.config:
                assert any(x.config.get('name', x.part) == v.config[part_type] for x in values['parts']), \
                    f'{v.config[part_type]} not found in the parts of {values["agent_name"]}'

        if isinstance(v, FunctorWrapper):
            cls._check_glue_parts(v.wrapped, values)
        elif isinstance(v, FunctorMultiWrapper):
            for elem in v.wrapped:
                cls._check_glue_parts(elem, values)
        elif isinstance(v, FunctorDictWrapper):
            for elem in v.wrapped.values():
                cls._check_glue_parts(elem, values)

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
        self.agent_glue_order: typing.Tuple = ()

        self._action_space_transformation: typing.Optional[SpaceTransformationBase] = None
        self._obs_space_transformation: typing.Optional[SpaceTransformationBase] = None

        # self._agent_glue_obs_export_behavior = {}

        # Sample parameter provider
        # This RNG only used here.  Normal use uses the one from the environment.
        rng, _ = gym.utils.seeding.np_random(0)
        self.fill_parameters(rng=rng, default_parameters=True)
        assert len(self.config.platform_names) == 1, "BaseAgent can only have one platform assigned to it currently"

    @property
    def platform_names(self) -> typing.List[str]:
        """
        Returns the name of the platform this agent is attached to

        Returns
        -------
        str:
            The name of the platform
        """
        return self.config.platform_names

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

        self.local_variable_store = flatten_dict.unflatten({k: v.get_value(rng, {}) for k, v in current_parameters.items()})

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

    def get_platform_parts(self, simulator, platform_dict):
        """
        Gets a list of the agent's platform parts.

        Parameters
        ----------
        simulator: Simulator class used by environment to simulate agent actions.
        platform_type: A mapping of platform_names to platform type enumeration corresponding to the agent's platform

        Returns
        -------
        list:
            List of platform parts
        """
        my_parts = {platform_name: [] for platform_name in self.config.platform_names}
        for platform_name in self.config.platform_names:
            mapped_platform = platform_dict[platform_name]
            for part in self.config.parts:
                tmp = PluginLibrary.FindMatch(part.part, {"simulator": simulator, "platform_type": mapped_platform})
                for ref_name, ref_key in part.references.items():
                    if ref_key not in self.config.reference_store:
                        raise RuntimeError(f'Part reference {ref_key} must be in the agent reference store.')
                    ref_value = self.config.reference_store[ref_key]
                    if isinstance(ref_value, Parameter):
                        raise TypeError(f'Part reference {ref_key} cannot be Parameter')
                    part.config[ref_name] = ref_value
                my_parts[platform_name].append((tmp, part.config))
        return my_parts

    def make_glues(self, platforms, agent_id: str, env_ref_stores: typing.List[typing.Dict[str, typing.Any]]) -> None:
        """
        Creates agent glue functors from agent configuration.

        Parameters
        ----------
        platforms: The platform_name to platform instances mapping associated with the glue functors.
        agent_id: The id of the agent associated with the glue functors.
        env_ref_stores: Reference stores for items managed by the environment

        Returns
        -------
        None
        """
        sort_order = {}
        # base agent only has one platform
        platform_name = self.config.platform_names[0]
        platform = platforms[platform_name]
        self.agent_glue_dict.clear()
        for glue_dict in self.config.glues:
            if self.config.normalization_master_switch == "ALL_OFF":
                glue_dict.config["normalization"] = {"enabled": False}
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

            # find dependent nodes for the topological search
            dependent_glues = {extractor.fields[0] for extractor in created_glue.config.extractors.values()}
            sort_order[glue_name] = dependent_glues

            # add the glue to the agent glue dict
            # self._agent_glue_obs_export_behavior[glue_name] = glue.training_obs_behavior
            self.agent_glue_dict[glue_name] = created_glue
        self.agent_glue_order = tuple(TopologicalSorter(sort_order).static_order())

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
                    platform_names=self.platform_names,
                    param_sources=[self.local_variable_store.get('rewards', {})],
                    ref_sources=[self.local_variable_store.get('reference_store', {}), self.config.reference_store] + env_ref_stores
                )
            )
        self.agent_reward_dict = RewardDict(processing_funcs=tmp)

    def make_platform_dones(
        self,
        env_params: typing.Dict[str, typing.List[typing.Sequence[typing.Dict[str, typing.Any]]]],
        env_ref_stores: typing.List[typing.Dict[str, typing.Any]]
    ) -> typing.Dict[str, typing.List[Functor]]:
        """
        Creates agent done functors from agent configuration.

        Parameters
        ----------
        agent_id: The id of the agent associated with the done functors.
        env_params: mapping of platform names to Parameters for the provided dones
        env_ref_stores: Reference stores for items managed by the environment

        Returns
        -------
        a dictionary containing mapping platform names to all dones the agent class wants
        to register to the platform
        """

        platform_dones: typing.Dict[str, typing.List[Functor]] = {}
        for platform_name in self.config.platform_names:
            my_platform_env_params = env_params[platform_name]
            platform_dones[platform_name] = []
            for done_dict in self.config.dones:
                platform_dones[platform_name].append(
                    done_dict.create_functor_object(
                        param_sources=[self.local_variable_store.get('dones', {})] + my_platform_env_params,
                        ref_sources=[self.local_variable_store.get('reference_store', {}), self.config.reference_store] + env_ref_stores,
                        agent_name=self.config.agent_name,
                        platform_name=platform_name
                    )
                )
        return platform_dones

    def _create_space(self, space_getter: typing.Optional[typing.Callable] = None):
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
        for glue_name in self.agent_glue_order:
            # call our space getter to pick which space we want,
            # for example: action_space, observation_space, normalized_action_space, normalized_observation_space
            space_def = space_getter(self.agent_glue_dict[glue_name])
            # if the space is None don't add it
            if space_def:
                return_space.spaces[glue_name] = space_def
        return return_space if len(return_space.spaces) > 0 else None

    @lru_cache(maxsize=1)
    def observation_space(self):
        """
        Returns agents obervation space
        """
        return self._create_space(space_getter=lambda glue_obj: glue_obj.observation_space())

    @lru_cache(maxsize=1)
    def normalized_observation_space(self):
        """
        Returns agents normalized obervation space
        """

        normalized_obs_space = self._create_space(
            space_getter=lambda glue_obj: glue_obj.normalized_observation_space()
            if glue_obj.config.training_export_behavior == TrainingExportBehavior.INCLUDE else None
        )

        if self.config.policy_observation_transformation:
            self._obs_space_transformation = self.config.policy_observation_transformation.transformation(
                normalized_obs_space, self.config.policy_observation_transformation.config
            )
            return self._obs_space_transformation.output_space

        return normalized_obs_space

    @lru_cache(maxsize=1)
    def observation_units(self):
        """
        Returns agents obervation units
        """
        return self._create_space(
            space_getter=lambda glue_obj: glue_obj.observation_units() if hasattr(glue_obj, "observation_units") else None
        )

    @lru_cache(maxsize=1)
    def action_space(self):
        """
        Returns agents action space
        """
        return self._create_space(space_getter=lambda glue_obj: glue_obj.action_space())

    @lru_cache(maxsize=1)
    def normalized_action_space(self):
        """
        Returns agents normalized action space
        """
        normalized_action_space = self._create_space(space_getter=lambda glue_obj: glue_obj.normalized_action_space())

        if self.config.policy_action_transformation:
            self._action_space_transformation = self.config.policy_action_transformation.transformation(
                normalized_action_space, self.config.policy_action_transformation.config
            )
            return self._action_space_transformation.output_space

        return normalized_action_space

    def apply_action(self, action: typing.Union[typing.Dict[typing.Any, typing.Any], typing.Any]):
        """
        Applies actions to agent.

        Parameters
        ----------
        action (optional): Either a dictionary of actions to be applied to agent, or
            a flattened action.

        Returns
        -------
        None
        """

        if self._action_space_transformation:
            action = self._action_space_transformation.convert_transformed_sample(action)

        raw_action_dict = collections.OrderedDict()
        obs = self.get_observations()
        for glue_name, normalized_action in action.items():
            glue_object = self.agent_glue_dict[glue_name]
            raw_action = glue_object.unnormalize_action(normalized_action)
            raw_action_dict[glue_name] = raw_action
            glue_object.apply_action(raw_action, obs, self.action_space(), self.observation_space(), self.observation_units())
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

    def get_observations(self):
        """
        Gets combined observation from agent glues.

        Returns
        -------
        OrderedDict
            A dictionary of glue observations in the form {glue_name: glue_observation}
        """
        return_observation: collections.OrderedDict = collections.OrderedDict()
        for glue_name in self.agent_glue_order:
            glue_object = self.agent_glue_dict[glue_name]
            glue_obs = glue_object.get_observation(return_observation, self.observation_space(), self.observation_units())
            if glue_obs:
                return_observation[glue_name] = glue_obs
                if glue_object.config.clip_to_space:
                    return_observation[glue_name] = EnvSpaceUtil.clip_space_sample_to_space(glue_obs, glue_object.observation_space())
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
        for glue_name in self.agent_glue_order:
            glue_info = self.agent_glue_dict[glue_name].get_info_dict()
            if glue_info:
                return_info[glue_name] = glue_info
        return return_info

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
            agent_id: name of the agent
            state: current simulator state
            batch: batch from the current trajectory
            episode: the episode object
            policy: the policy that ran the trajectory
            reward_info: the info dict for rewards
        """
        for reward_glue in self.agent_reward_dict.process_callbacks:
            output_value = reward_glue.post_process_trajectory(agent_id, state, batch, episode, policy)
            if output_value is not None:

                # If we have a bad setup the agent may be "dead" on first step
                # This in turn causes major issues when trying to index reward_info
                # Check that the agent exists
                if agent_id in reward_info:
                    reward_info[agent_id].setdefault(reward_glue.name, 0.0)
                    reward_info[agent_id][reward_glue.name] += output_value

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

    def set_removed(self, removed_state: typing.Dict[str, bool]):
        """
        Set the agent_removed flag for the agent's glues.

        Parameters
        ----------
        - removed_state (typing.Dict[str, bool]): The inoperability status of the platform an agent class is controlling.
            True: the platform is no longer operable
            False: the platform is still operable

        Returns
        -------
        None
        """
        if not removed_state[self.platform_names[0]]:
            for glue in self.agent_glue_dict.values():
                glue.set_agent_removed(True)

    def create_training_observations(self, observations: collections.OrderedDict):
        """
        Base class representing a trainable agent in an environment.
        """

        training_obs = collections.OrderedDict()
        for obs_name, obs in observations.items():
            glue_obj = self.get_glue(obs_name)
            if glue_obj is not None and glue_obj.config.training_export_behavior == TrainingExportBehavior.INCLUDE:
                training_obs[obs_name] = self.normalize_observation(obs_name, obs)

        if self._obs_space_transformation:
            return self._obs_space_transformation.convert_sample(sample=training_obs)

        return training_obs


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
