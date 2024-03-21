"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Base Agent Class
"""
from __future__ import annotations

import abc
import collections
import copy
import fractions
import typing
from collections.abc import MutableMapping
from functools import cached_property
from graphlib import TopologicalSorter
from typing import Any, Literal

import flatten_dict
import gymnasium
from pydantic import BaseModel, ConfigDict, Field, ImportString, field_validator
from tree import map_structure

from corl.dones.done_func_base import DoneFuncBase
from corl.dones.episode_length_done import EpisodeLengthDone
from corl.episode_parameter_providers import EpisodeParameterProvider, Randomness
from corl.glues.base_glue import BaseAgentGlue, TrainingExportBehavior
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.factory import Factory
from corl.libraries.functor import Functor, FunctorDictWrapper, FunctorMultiWrapper, FunctorWrapper, ObjectStoreElem
from corl.libraries.parameters import Parameter
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import DictProp, NestedDict, Prop
from corl.libraries.space_transformations import SpaceTransformationBase
from corl.libraries.units import corl_get_ureg
from corl.rewards.reward_func_base import RewardFuncBase
from corl.simulators.base_simulator import validation_helper_units_and_parameters


class PartsFunctor(BaseModel):
    """
    - part: The name of the part. This should be registered to a corresponding platform part class in the plugin library.
    - config: The specific configuration dictionary expected by the platform part.
    """

    part: str
    config: dict[str, Any] = {}
    references: dict[str, str] = {}


class AgentParseBase(BaseModel):
    """
    - agent: The BaseAgent class representing the agent.
    - config: The agent specific configuration including glues, reward, dones, parts, etc.
    """

    agent: ImportString
    config: dict[str, Any]

    @field_validator("agent")
    @classmethod
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
    platform_names: list[str]
    policy_config: dict[str, Any]


class PlatformParseInfo(BaseModel):
    """
    - platform_config
    """

    platform_config: dict[str, Any]


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

    glues: dict[str, dict[str, typing.Any]] = {}
    rewards: dict[str, dict[str, typing.Any]] = {}
    dones: dict[str, dict[str, typing.Any]] = {}
    reference_store: dict[str, typing.Any] = {}
    simulator_reset: dict[str, typing.Any] = {}

    @staticmethod
    def _validate_leaves_are_parameters(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                BaseAgentEppParameters._validate_leaves_are_parameters(value)
        elif not isinstance(obj, Parameter):
            raise TypeError(f"Invalid type: {type(obj)} (required type: {Parameter.__qualname__})")

    @field_validator("glues", "rewards", "dones", "reference_store", "simulator_reset")
    @classmethod
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

    transformation: ImportString
    config: dict = {}

    @field_validator("transformation")
    @classmethod
    def validate_transformation(cls, val):
        """
        validates transformation subclass type
        """
        if issubclass(val, SpaceTransformationBase):
            return val

        raise TypeError(f"Invalid output_type {type(val)} in SpaceTypeConversionValidator")


def check_rewards(v):
    """Check if rewards subclass RewardFuncBase"""
    if not issubclass(v.functor, RewardFuncBase):
        raise TypeError(f"Reward functors must subclass RewardFuncBase, but reward {v.name} is of type {v.functor}")
    return v


def check_dones(v):
    """Check if dones subclass DoneFuncBase"""
    if not issubclass(v.functor, DoneFuncBase):
        raise TypeError(f"Done functors must subclass DoneFuncBase, but done {v.name} is of type {v.functor}")
    if issubclass(v.functor, EpisodeLengthDone):
        raise ValueError("Cannot specify EpisodeLengthDone as it is automatically added")
    return v


class BaseAgentParser(BaseModel):
    """
    - parts: A list of part configurations.
    - glues: A list of glue functor configurations.
    - rewards: A optional list of reward functor configurations
    - dones: An optional list of done functor configurations
    """

    # these can be sensors, or controller
    agent_name: str
    parts: list[list[PartsFunctor]]  # this uses plugin loader to load
    reference_store: dict[str, ObjectStoreElem] = {}
    glues: list[list[FunctorMultiWrapper | (FunctorWrapper | (Functor | FunctorDictWrapper))]]
    rewards: list[FunctorMultiWrapper | (FunctorWrapper | (Functor | FunctorDictWrapper))] = []
    dones: list[FunctorMultiWrapper | (FunctorWrapper | (Functor | FunctorDictWrapper))] = Field(default=[], validate_default=True)
    simulator_reset_parameters: dict[str, Any] = {}
    # frame_rate in Hz (i.e. .25 indicates this agent will process when sim_tim % 4 == 0)
    frame_rate: float = 1.0
    platform_names: list[str]
    multiple_workers: bool = False
    policy_action_transformation: AgentSpaceTransformation | None = None
    policy_observation_transformation: AgentSpaceTransformation | None = None
    # modifies the normalization for any glue created by this agent
    normalization_master_switch: Literal["ALL_ON", "DEFAULT", "ALL_OFF"] = "DEFAULT"
    episode_parameter_provider: Factory
    episode_parameter_provider_parameters: BaseAgentEppParameters = Field(validate_default=True, default=None)  # type: ignore
    epp: EpisodeParameterProvider = Field(validate_default=True, default=None)  # type: ignore
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("parts", "glues", mode="before")
    @classmethod
    def convert_parts_and_glues(cls, v):
        """Check if parts subclass BaseAgentGlue"""
        if isinstance(v, list):
            if all(isinstance(elem, list) for elem in v):
                return v
            return [v]
        raise TypeError(f"Parts and glues must be List not {type(v)}")

    @field_validator("glues")
    def check_glues(cls, v, info):
        """Check if glues subclass BaseAgentGlue"""
        for glue_set in v:
            for glue in glue_set:
                if not issubclass(glue.functor, BaseAgentGlue):
                    raise TypeError(f"Glue functors must subclass BaseAgentGlue, but glue {v.name} is of type {v.functor}")

        assert "agent_name" in info.data
        assert "parts" in info.data

        for glues, parts in zip(v, info.data["parts"]):
            for glue in glues:
                cls._check_glue_parts(glue, parts, info.data)

        return v

    @classmethod
    def _check_glue_parts(cls, v, parts, values) -> None:
        for part_type in ["sensor", "controller"]:
            if part_type in v.config:
                assert any(
                    x.config.get("name", x.part) == v.config[part_type] for x in parts
                ), f'{v.config[part_type]} not found in the parts of {values["agent_name"]}'

        if isinstance(v, FunctorWrapper):
            cls._check_glue_parts(v.wrapped, parts, values)
        elif isinstance(v, FunctorMultiWrapper):
            for elem in v.wrapped:
                cls._check_glue_parts(elem, parts, values)
        elif isinstance(v, FunctorDictWrapper):
            for elem in v.wrapped.values():
                cls._check_glue_parts(elem, parts, values)

    @field_validator("simulator_reset_parameters", mode="before")
    @classmethod
    def update_units_and_parameters(cls, v):
        """Update simulation reset parameters to meet base simulator requirements."""
        return validation_helper_units_and_parameters(v)

    @field_validator("episode_parameter_provider")
    def check_epp(cls, epp_factory, values):
        """Check if episode parameter provider subclass EpisodeParameterProvider"""
        if not issubclass(epp_factory.type, EpisodeParameterProvider):
            raise TypeError(f"Episode parameter providers must subclass EpisodeParameterProvider, but is is of type {epp_factory.type}")

        # replace %%AGENT%% with agent_name
        epp_params = flatten_dict.flatten(epp_factory.config)
        for key, value in epp_params.items():
            if isinstance(value, str):
                epp_params[key] = value.replace(r"%%AGENT%%", values.data["agent_name"])
        epp_factory.config = flatten_dict.unflatten(epp_params)

        return epp_factory

    @field_validator("episode_parameter_provider_parameters", mode="before")
    def build_episode_parameter_provider_parameters(cls, _v, values) -> BaseAgentEppParameters:
        """Create the episode parameter provider config"""

        for key in ["reference_store", "glues", "rewards", "dones", "simulator_reset_parameters"]:
            assert key in values.data

        reference_parameters: dict[str, Parameter] = {
            ref_name: ref_value for ref_name, ref_value in values.data["reference_store"].items() if isinstance(ref_value, Parameter)
        }
        glue_parameters: dict[str, dict[str, Parameter]] = {}
        for glue_set in values.data["glues"]:
            for functor in glue_set:
                functor.add_to_parameter_store(glue_parameters)

        reward_parameters: dict[str, dict[str, Parameter]] = {}
        for functor in values.data["rewards"]:
            functor.add_to_parameter_store(reward_parameters)

        done_parameters: dict[str, dict[str, Parameter]] = {}
        for functor in values.data["dones"]:
            functor.add_to_parameter_store(done_parameters)

        flat_data = flatten_dict.flatten(values.data["simulator_reset_parameters"])
        for key, value in flat_data.items():
            if isinstance(value, BaseModel):
                flat_data[key] = value.dict()
        expanded_sim_reset_params = flatten_dict.unflatten(flat_data)

        sim_parameters_flat = {
            name: param for name, param in flatten_dict.flatten(expanded_sim_reset_params).items() if isinstance(param, Parameter)
        }
        sim_parameters = flatten_dict.unflatten(sim_parameters_flat)

        return BaseAgentEppParameters(
            glues=glue_parameters,
            rewards=reward_parameters,
            dones=done_parameters,
            reference_store=reference_parameters,
            simulator_reset=sim_parameters,
        )

    @field_validator("epp", mode="before")
    def build_epp(cls, epp, values):
        """Builds an instance of an EpisodeParameterProvider if necessary"""
        if epp is None:
            assert "episode_parameter_provider_parameters" in values.data

            epp_params = dict(values.data["episode_parameter_provider_parameters"])
            flat_epp_params = flatten_dict.flatten(epp_params)
            epp = values.data["episode_parameter_provider"].build(parameters=flat_epp_params)
        return epp

    @field_validator("frame_rate")
    def simplify_frame_rate(cls, v):
        """Expand the precision of the frame rate (e.g. .333 -> .3333333333334)"""
        f = fractions.Fraction(v).limit_denominator(20)
        return f.numerator / f.denominator


class BaseAgent:  # noqa: PLR0904
    """
    Base class representing an agent in an environment.
    """

    def __init__(self, **kwargs) -> None:
        self.config: BaseAgentParser = self.get_validator()(**kwargs)
        self.agent_glue_dict: dict[str, BaseAgentGlue] = {}
        self.agent_reward_functors: list[RewardFuncBase] = []
        self.agent_glue_order: tuple = ()

        self._action_space_transformation: SpaceTransformationBase | None = None
        self._obs_space_transformation: SpaceTransformationBase | None = None

        # self._agent_glue_obs_export_behavior = {}

        # Sample parameter provider
        # This RNG only used here.  Normal use uses the one from the environment.
        rng, _ = gymnasium.utils.seeding.np_random(0)
        self.fill_parameters(rng=rng, default_parameters=True)
        assert len(self.config.platform_names) == 1, "BaseAgent can only have one platform assigned to it currently"

    @property
    def platform_names(self) -> list[str]:
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

    @staticmethod
    def get_validator() -> type[BaseAgentParser]:
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

    def fill_parameters(self, rng: Randomness, env_epp_ctx: dict | None = None, default_parameters: bool = False) -> None:
        """Sample the episode parameter provider to fill the local variable store."""
        if default_parameters:
            current_parameters = self.config.epp.config.parameters
        else:
            current_parameters, _, _ = self.config.epp.get_params(rng, env_epp_ctx)

        flat_variable_store = {}
        for k, v in current_parameters.items():
            param_value = v.get_value(rng, {})
            if isinstance(param_value, dict):
                for param_k, param_v in param_value.items():
                    # for the dict case the param has a path that would be something like
                    # ('simulator_reset', 'platforms', 'paddle0', 'y_and_vel')
                    # we want to collapse down to the nested params so
                    # ('simulator_reset', 'platforms', 'paddle0', 'y')
                    # and ('simulator_reset', 'platforms', 'paddle0', 'vel')
                    # this may be evil and need to be reworked, but not 100% sure
                    flat_variable_store[(*k[:-1], param_k)] = param_v
            else:
                flat_variable_store[k] = param_value

        self.local_variable_store = flatten_dict.unflatten(flat_variable_store)

    def get_simulator_reset_parameters(self) -> dict[str, typing.Any]:
        """Return the local parameters needed within the simulator reset"""
        output = copy.deepcopy(self.config.simulator_reset_parameters)

        def expand_and_update(data: MutableMapping[str, Any], simulator_reset: typing.Mapping[str, Any]):
            for key, value in data.items():
                if isinstance(value, BaseModel):
                    expand_and_update(vars(value), simulator_reset.get(key, {}))
                elif isinstance(value, MutableMapping):
                    expand_and_update(value, simulator_reset.get(key, {}))
                elif key in simulator_reset:
                    data[key] = simulator_reset[key]

        expand_and_update(output, self.local_variable_store.get("simulator_reset", {}))

        return output

    def get_platform_parts(self, simulator, platform_dict) -> dict[str, list]:
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
        my_parts: dict[str, list] = {platform_name: [] for platform_name in self.config.platform_names}
        for platform_name, platform_parts in zip(self.config.platform_names, self.config.parts):
            mapped_platform = platform_dict[platform_name]
            for part in platform_parts:
                tmp = PluginLibrary.FindMatch(part.part, {"simulator": simulator, "platform_type": mapped_platform})
                for ref_name, ref_key in part.references.items():
                    if ref_key not in self.config.reference_store:
                        raise RuntimeError(f"Part reference {ref_key} must be in the agent reference store.")
                    ref_value = self.config.reference_store[ref_key]
                    if isinstance(ref_value, Parameter):
                        raise TypeError(f"Part reference {ref_key} cannot be Parameter")
                    part.config[ref_name] = ref_value
                my_parts[platform_name].append((tmp, part.config))
        return my_parts

    def make_glues(self, platforms, agent_id: str, env_ref_stores: list[dict[str, typing.Any]]) -> None:
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
        self.agent_glue_dict.clear()
        for config_glues, platform_name in zip(self.config.glues, self.config.platform_names):
            platform = platforms[platform_name]
            for glue_dict in config_glues:
                if self.config.normalization_master_switch == "ALL_OFF":
                    glue_dict.config["normalization"] = {"enabled": False}
                created_glue = glue_dict.create_functor_object(
                    platform=platform,
                    agent_name=agent_id,
                    param_sources=[self.local_variable_store.get("glues", {})],
                    ref_sources=[self.local_variable_store.get("reference_store", {}), self.config.reference_store, *env_ref_stores],
                )
                glue_name = created_glue.get_unique_name()
                if glue_name is None:
                    raise RuntimeError(f"No glue name for {created_glue}")

                if glue_name in self.agent_glue_dict:
                    raise RuntimeError(f"The {glue_name} glue has a unique name, but it already exists")
                # find dependent nodes for the topological search
                dependent_glues = {extractor.fields[0] for extractor in created_glue.config.extractors.values()}
                sort_order[glue_name] = dependent_glues

                self.agent_glue_dict[glue_name] = created_glue
        self.agent_glue_order = tuple(TopologicalSorter(sort_order).static_order())

    def make_rewards(self, agent_id: str, env_ref_stores: list[dict[str, typing.Any]]) -> None:
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
        tmp = [
            reward_dict.create_functor_object(
                agent_name=agent_id,
                platform_names=self.platform_names,
                param_sources=[self.local_variable_store.get("rewards", {})],
                ref_sources=[self.local_variable_store.get("reference_store", {}), self.config.reference_store, *env_ref_stores],
            )
            for reward_dict in self.config.rewards
        ]
        self.agent_reward_functors = tmp

    def make_platform_dones(
        self, env_params: dict[str, list[typing.Sequence[dict[str, typing.Any]]]], env_ref_stores: list[dict[str, typing.Any]]
    ) -> dict[str, list[DoneFuncBase]]:
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

        platform_dones: dict[str, list[DoneFuncBase]] = {}
        for platform_name in self.config.platform_names:
            my_platform_env_params = env_params[platform_name]
            platform_dones[platform_name] = []
            for done_dict in self.config.dones:
                platform_dones[platform_name].append(
                    done_dict.create_functor_object(
                        param_sources=[self.local_variable_store.get("dones", {}), *my_platform_env_params],  # type: ignore[list-item]
                        ref_sources=[self.local_variable_store.get("reference_store", {}), self.config.reference_store, *env_ref_stores],
                        agent_name=self.config.agent_name,
                        platform_name=platform_name,
                    )
                )
        return platform_dones

    def __create_space(
        self, return_space: dict | gymnasium.spaces.dict.Dict, space_getter: typing.Callable | None = None
    ) -> dict | gymnasium.spaces.dict.Dict | None:
        """
        Creates a gymnasium dict space from the agent's glues.

        Parameters
        ----------
        return_space typing.Union[typing.Dict, gymnasium.spaces.dict.Dict]: empty dict or space
        space_getter (optional): A function that takes a glue_obj: BaseAgentGlue and returns a space.
            Default is space_getter=lambda glue_obj: glue_obj.observation_space.

        Returns
        -------
        typing.Optional[typing.Union[typing.Dict, gymnasium.spaces.dict.Dict]]
            A gymnasium dict space or dict composed of all the spaces returned by applying the space_getter to the agent glues.
        """
        if space_getter is None:

            def default_getter(glue_obj):
                return glue_obj.observation_space

            space_getter = default_getter

        # loop over all glue name and  glue_obj pairs
        for glue_name in self.agent_glue_order:
            # call our space getter to pick which space we want,
            # for example: action_space, observation_space, normalized_action_space, normalized_observation_space
            space_def = space_getter(self.agent_glue_dict[glue_name])
            # if the space is None don't add it
            if space_def:
                return_space[glue_name] = space_def
        return return_space if len(return_space) > 0 else None

    def _create_space_dict(self, space_getter: typing.Callable | None = None):
        """
        Creates a dict from the agent's glues.

        Parameters
        ----------
        space_getter (optional): A function that takes a glue_obj: BaseAgentGlue and returns a space.
            Default is space_getter=lambda glue_obj: glue_obj.observation_space.

        Returns
        -------
        dict()
            A dict composed of all the spaces returned by applying the space_getter to the agent glues.
        """
        return self.__create_space({}, space_getter)

    def _create_space(self, space_getter: typing.Callable | None = None):
        """
        Creates a gymnasium dict space from the agent's glues.

        Parameters
        ----------
        space_getter (optional): A function that takes a glue_obj: BaseAgentGlue and returns a space.
            Default is space_getter=lambda glue_obj: glue_obj.observation_space.

        Returns
        -------
        gymnasium.spaces.dict.Dict()
            A gymnasium dict space composed of all the spaces returned by applying the space_getter to the agent glues.
        """
        return self.__create_space(gymnasium.spaces.dict.Dict(), space_getter)

    @cached_property
    def observation_prop(self) -> Prop:
        """
        Returns agents prop
        """
        return DictProp(spaces=self._create_space_dict(space_getter=lambda glue_obj: glue_obj.observation_prop))

    @cached_property
    def observation_units(self):
        """
        Returns agents normalized action space
        """
        return self.observation_prop.get_units()

    @cached_property
    def observation_space(self) -> gymnasium.Space:
        """
        Returns agents observation space
        """
        # calling it this way instead of getting it from the prop
        # causes the obs space cache to get hit on each glue instead of
        # recreating it from scratch using self.observation_prop.create_space()
        return self._create_space(space_getter=lambda glue_obj: glue_obj.observation_space)

    @cached_property
    def action_prop(self) -> Prop:
        """
        Returns the action prop that defines this agents actions
        """
        return DictProp(spaces=self._create_space_dict(space_getter=lambda glue_obj: glue_obj.action_prop))

    @cached_property
    def action_units(self) -> str | NestedDict:
        """
        Returns a nested dictionary where each leaf node is
        the unit of the observation matching the same access chain in
        the action
        """
        return self.action_prop.get_units()

    @cached_property
    def action_space(self) -> gymnasium.Space:
        """
        Returns agents action space
        """
        # calling it this way instead of getting it from the prop
        # causes the action space cache to get hit on each glue instead of
        # recreating it from scratch using self.action_prop.create_space()
        return self._create_space(space_getter=lambda glue_obj: glue_obj.action_space)

    @cached_property
    def normalized_observation_space(self) -> gymnasium.spaces.Space:
        """
        Returns agents normalized observation space
        """
        normalized_obs_space = self._create_space(
            space_getter=lambda glue_obj: glue_obj.normalized_observation_space
            if glue_obj.config.training_export_behavior == TrainingExportBehavior.INCLUDE
            else None
        )

        self._obs_space_transformation = self.__create_obs_space_transformation(normalized_obs_space)
        if self._obs_space_transformation:
            return self._obs_space_transformation.output_space
        return normalized_obs_space

    @cached_property
    def normalized_action_space(self):
        """
        Returns agents normalized action space
        """
        norm_space = self._create_space(space_getter=lambda glue_obj: glue_obj.normalized_action_space)

        self._action_space_transformation = self.__create_action_space_transformation(norm_space)

        if self._action_space_transformation:
            return self._action_space_transformation.output_space

        return norm_space

    def __create_action_space_transformation(self, space):
        """Create action space transformation"""
        if self.config.policy_action_transformation:
            return self.config.policy_action_transformation.transformation(space, self.config.policy_action_transformation.config)
        return None

    def __create_obs_space_transformation(self, space):
        """Create observations space transformation"""
        if self.config.policy_observation_transformation:
            return self.config.policy_observation_transformation.transformation(space, self.config.policy_observation_transformation.config)
        return None

    def apply_action(self, action: dict[typing.Any, typing.Any] | typing.Any, last_obs: typing.OrderedDict):
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
        for glue_name, normalized_action in action.items():
            glue_object = self.agent_glue_dict[glue_name]
            raw_action = glue_object.unnormalize_action(normalized_action)
            raw_action = map_structure(lambda x, y: corl_get_ureg().Quantity(x, y), raw_action, self.action_units[glue_name])
            raw_action_dict[glue_name] = raw_action
            glue_object.apply_action(raw_action, last_obs, self.action_space, self.observation_space, self.observation_units)
        return raw_action_dict

    def create_next_action(self, observation: dict, reward: float, done: bool, info: dict):
        """
        Creates next action agent will apply.

        Parameters
        ----------
        action_dict (optional): A dictionary of actions to be applied to agent.

        Returns
        -------
        None
        """

    def get_observations(self) -> collections.OrderedDict:
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
            if glue_obs := glue_object.get_observation(
                return_observation,
                self.observation_space,
                self.observation_units,
            ):
                return_observation[glue_name] = glue_obs
                if glue_object.config.clip_to_space and glue_object.observation_space:
                    return_observation[glue_name] = EnvSpaceUtil.clip_space_sample_to_space(glue_obs, glue_object.observation_space)

        return return_observation

    def get_info_dict(self) -> collections.OrderedDict:
        """
        Gets combined observation from agent glues.

        Returns
        -------
        OrderedDict
            A dictionary of glue observations in the form {glue_name: glue_observation}
        """
        return_info: collections.OrderedDict = collections.OrderedDict()
        for glue_name in self.agent_glue_order:
            if glue_info := self.agent_glue_dict[glue_name].get_info_dict():
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
        {agent_id: float}, {agent_id: Dict[str, float]}
        """
        reward_info = {}
        reward_total = 0.0
        for reward_functor in self.agent_reward_functors:
            output_value = reward_functor(
                observation=observation,
                action=action,
                next_observation=next_observation,
                state=state,
                next_state=next_state,
                observation_space=observation_space,
                observation_units=observation_units,
            )
            reward_total += output_value
            reward_info[reward_functor.name] = output_value

        return reward_total, reward_info

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
        for reward_functor in self.agent_reward_functors:
            output_value = reward_functor.post_process_trajectory(agent_id, state, batch, episode, policy)
            # If we have a bad setup the agent may be "dead" on first step
            # This in turn causes major issues when trying to index reward_info
            # Check that the agent exists
            if output_value is not None and agent_id in reward_info:
                reward_info[agent_id].setdefault(reward_functor.name, 0.0)
                reward_info[agent_id][reward_functor.name] += output_value

    def get_glue(self, name: str) -> BaseAgentGlue | None:
        """
        Get the glue object with the given name
        """
        return self.agent_glue_dict[name] if name in self.agent_glue_dict else None

    def normalize_observation(self, obs_name: str, obs: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type | None:
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
        return glue_obj.normalize_observation(obs) if glue_obj is not None else None

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
            if normalized_obs := self.normalize_observation(obs_name, obs):
                normalized_observation_dict[obs_name] = normalized_obs

        return normalized_observation_dict

    def set_removed(self, removed_state: dict[str, bool]):
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

    def reset(self) -> None:
        """
        Method to allow child classes to call reset-dependent
        functionality, for instance at the end of an episode.
        """


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


class NonTrainableBaseAgent(BaseAgent):
    """
    Base class representing a non-trainable agent in an environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    @abc.abstractmethod
    def create_action(self, observation: dict):
        """
        Creates an action given observation.

        Parameters
        ----------
        observation: dict
            agent's observation of the current environment

        Returns
        -------
        OrderedDict[str: Any]
            A dictionary of non-normalized actions applied to glues in
            the form {glue_name: raw_action}
        """
