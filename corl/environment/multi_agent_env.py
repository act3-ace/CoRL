"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import contextlib
import copy
import fractions
import json
import logging
import math
import os
import pickle
import sys
import typing
import warnings
from collections import OrderedDict, defaultdict
from functools import partial
from graphlib import TopologicalSorter
from itertools import chain
from json import JSONEncoder, dumps, loads
from pathlib import Path
from typing import Annotated

import deepmerge
import flatten_dict
import gymnasium.spaces
import gymnasium.utils.seeding
import numpy as np
import tree
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    TypeAdapter,
    field_validator,
)

from corl.agents.base_agent import AgentParseInfo
from corl.dones.done_func_base import DoneFuncBase, SharedDoneFuncBase
from corl.dones.episode_length_done import EpisodeLengthDone
from corl.environment.base_multi_agent_env import BaseCorlMultiAgentEnv, BaseCorlMultiAgentEnvValidator
from corl.environment.utils import env_creation
from corl.environment.utils.obs_buffer import ObsBuffer
from corl.environment.utils.space_sort import gymnasium_space_sort
from corl.episode_parameter_providers import EpisodeParameterProvider
from corl.libraries.collection_utils import get_dictionary_subset
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.environment_dict import DoneDict
from corl.libraries.factory import Factory
from corl.libraries.functor import Functor, ObjectStoreElem
from corl.libraries.parameters import ConstantParameter, Parameter, ParameterWrapper
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import DictProp
from corl.libraries.units import Quantity, corl_quantity
from corl.libraries.utils import replace_magic_strings
from corl.simulators.base_platform import BasePlatform
from corl.simulators.base_simulator import BaseSimulator, BaseSimulatorState, validation_helper_units_and_parameters

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def check_done(v) -> None:
    """Check if dones subclass DoneFuncBase"""
    if not issubclass(v.functor, DoneFuncBase):
        raise TypeError(f"Done functors must subclass DoneFuncBase, but done {v.name} is of type {v.functor}")
    if issubclass(v.functor, EpisodeLengthDone):
        raise ValueError("Cannot specify EpisodeLengthDone as it is automatically added")
    return v


def check_shared(v):
    """Check if dones subclass SharedDoneFuncBase"""
    if not issubclass(v.functor, SharedDoneFuncBase):
        raise TypeError(f"Shared Done functors must subclass SharedDoneFuncBase, but done {v.name} is of type {v.functor}")
    return v


class EnvironmentDoneValidator(BaseModel):
    """Validation model for the dones of ACT3MultiAgentEnv"""

    world: list[Annotated[Functor, AfterValidator(check_done)]] = []
    task: dict[str, list[Annotated[Functor, AfterValidator(check_done)]]] = {}
    shared: list[Annotated[Functor, AfterValidator(check_shared)]] = []


class ACT3MultiAgentEnvEppParameters(BaseModel):
    """
    world: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        keys: done name, parameter name

    task: typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]] = {}
        keys: agent name, done name, parameter name

    shared: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        keys: done name, parameter name

    reference_store: typing.Dict[str, typing.Any] = {}
        keys: reference name

    simulator_reset: typing.Dict[str, typing.Any] = {}
        keys: whatever the simulator wants, but it needs to be kwargs to simulator reset
    """

    world: dict[str, dict[str, typing.Any]] = {}
    task: dict[str, dict[str, dict[str, typing.Any]]] = {}
    shared: dict[str, dict[str, typing.Any]] = {}
    reference_store: dict[str, typing.Any] = {}
    simulator_reset: dict[str, typing.Any] = {}

    @staticmethod
    def _validate_leaves_are_parameters(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                ACT3MultiAgentEnvEppParameters._validate_leaves_are_parameters(value)
        elif not isinstance(obj, Parameter | ParameterWrapper):
            raise TypeError(f"Invalid type: {type(obj)} (required type: {Parameter.__qualname__})")

    @field_validator("world", "task", "shared", "reference_store", "simulator_reset")
    @classmethod
    def validate_leaves_are_parameters(cls, v):
        """
        checks to make sure outer most leaf nodes of config are parameters
        """
        ACT3MultiAgentEnvEppParameters._validate_leaves_are_parameters(v)
        return v


def resolve_reference_store_factory(v):
    """
    allow the user to short hand constant parameter to {"value": X, "units": Y}
    """
    if isinstance(v, dict) and "value" in v and ("units" in v or "unit" in v):
        v = {"type": ConstantParameter, "config": {"value": v["value"], "units": v.get("units", v.get("unit"))}}
    return Factory.resolve_factory(v, {})


class ACT3MultiAgentEnvValidator(BaseCorlMultiAgentEnvValidator):
    """Validation model for the inputs of ACT3MultiAgentEnv"""

    num_workers: NonNegativeInt = 0
    worker_index: NonNegativeInt = 0
    vector_index: NonNegativeInt | None = None
    remote: bool = False
    deep_sanity_check: bool = True
    save_state_pickle: bool = True

    horizon: PositiveInt = 1000
    sanity_check_obs: PositiveInt = 50
    sensors_grid: list | None = None
    plugin_paths: list[str] = []

    # Regex allows letters, numbers, underscore, dash, dot
    # Regex in output_path validator also allows forward slash
    # Empty string is not allowed
    TrialName: Annotated[str, StringConstraints(pattern="^[\\w\\.-]+$")] | None = None
    output_date_string: Annotated[str, StringConstraints(pattern="^[\\w\\.-]+$")] | None = None
    skip_pbs_date_update: bool = False
    # MyPy error ignored because it is handled by the pre-validator
    output_path: str = None  # type: ignore[assignment]

    agent_platforms: dict
    agents: dict[str, AgentParseInfo]

    simulator: Factory
    other_platforms: dict[str, dict[str, typing.Any]] = {}

    reference_store: dict[str, Annotated[ObjectStoreElem, BeforeValidator(resolve_reference_store_factory)]] = {}
    dones: EnvironmentDoneValidator = Field(default=EnvironmentDoneValidator(), validate_default=True)
    end_episode_on_first_agent_done: bool = False
    simulator_reset_parameters: dict[str, typing.Any] = {}

    episode_parameter_provider: Factory
    episode_parameter_provider_parameters: ACT3MultiAgentEnvEppParameters = Field(default=None, validate_default=True)  # type: ignore
    epp_registry: dict[str, EpisodeParameterProvider] = Field(default=None, validate_default=True)  # type: ignore

    max_agent_rate: int = 20  # the maximum rate (in Hz) that an agent may be run at
    timestep_epsilon: float = 1e-3
    sim_warmup_seconds: int = 0  # number of seconds to step simulator before getting initial obs

    # dict containing logging data useful to the external system, this will be sent to simulator::mark_episode_done
    metadata: dict = {}

    @property
    def epp(self) -> EpisodeParameterProvider:
        """
        return the current episode parameter provider
        """
        return self.epp_registry[ACT3MultiAgentEnv.episode_parameter_provider_name]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("plugin_paths")
    @classmethod
    def add_plugin_paths(cls, v):
        """Use the plugin path attribute to initialize the plugin library."""
        # @calong NOTES ON THIS CODE: DO NOT REMOVE the reraise
        # something about this code is strange when exceptions are raised and
        # either pydantic is eating the exception if something is not reraised (most likely)
        # or there is some dark magic causing exceptions to not happen, this code is a
        # simple way to make sure the exception is always raised and bypasses whatever
        # is going on, otherwise if an exception is raised the plugin process stops silently
        # and the env will fail to find a plugin and you will be left very confused
        try:
            PluginLibrary.add_paths(v)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("A plugin along your plugin paths failed to load due to the following error") from e
        return v

    @field_validator("output_path")
    def create_output_path(cls, v, values):
        """Build the output path."""

        v = v or "/tmp/data/act3/ray_results"  # noqa: S108
        adapter = TypeAdapter(Annotated[str, StringConstraints(pattern=r"^[\w/\.-]+$")])
        v = adapter.validate_python(v)

        if values.data["TrialName"] is not None:
            trial_prefix = (
                "" if values.data["skip_pbs_date_update"] else os.environ.get("PBS_JOBID", os.environ.get("TRIAL_NAME_PREFIX", ""))
            )

            v = os.path.join(v, f"{trial_prefix}-{values.data['TrialName']}") if trial_prefix else os.path.join(v, values.data["TrialName"])

        if values.data["output_date_string"] is not None and not values.data["skip_pbs_date_update"]:
            v = os.path.join(v, values.data["output_date_string"])

        v = os.path.abspath(v)
        v = os.path.join(v, str(values.data["worker_index"]).zfill(4))
        if values.data["vector_index"] is not None:
            v = os.path.join(v, str(values.data["vector_index"]).zfill(4))
        os.makedirs(v, exist_ok=True)

        return v

    @field_validator("simulator", mode="before")
    @classmethod
    def resolve_simulator_plugin(cls, v):
        """Determine the simulator from the plugin library."""
        assert isinstance(v, dict)
        try:
            v["type"]
        except (TypeError, KeyError):
            # Let pydantic print out an error when there is no type field
            return v

        match = PluginLibrary.FindMatch(v["type"], {})
        if not issubclass(match, BaseSimulator):
            raise TypeError(f"Simulator must subclass BaseSimulator, but is is of type {v['type']}")

        return {"type": match, "config": v.get("config")}

    @field_validator("agents")
    def agents_not_empty(cls, v, values):
        """Ensure that at least one agent exists"""
        if len(v) == 0:
            raise RuntimeError("No agents exist")

        agent_platforms = set(values.data["agent_platforms"])
        for agent_name, agent in v.items():
            if not all(platform_name in agent_platforms for platform_name in agent.platform_names):
                raise RuntimeError(
                    f"Agent {agent_name} is looking for platforms {agent.platform_names} "
                    f"but the only platforms available are {agent_platforms}"
                )

            if any(agent_name == platform_name for platform_name in agent.platform_names):
                raise RuntimeError(
                    f"agent '{agent_name}' has the same name as one of it's platforms {agent.platform_names}.\n"
                    " This leads to easily misconfigured code and is therefore not allowed."
                )
        return v

    @field_validator("dones")
    def agents_match(cls, v, values):
        """Ensure that platform in task dones match provided platforms"""
        # No extra agents in task dones
        for platform in v.task:
            if platform not in values.data["agent_platforms"]:
                raise RuntimeError(f"Platform {platform} lists a done condition but is not an allowed platform")

        # Task dones exist for all agents.  Make empty ones if necessary
        for platform in values.data["agent_platforms"]:
            if platform not in v.task:
                v.task[platform] = {}

        return v

    @field_validator("simulator_reset_parameters", mode="before")
    @classmethod
    def update_units_and_parameters(cls, v):
        """Update simulation reset parameters to meet base simulator requirements."""
        return validation_helper_units_and_parameters(v)

    @field_validator("episode_parameter_provider_parameters", mode="before")
    def build_episode_parameter_provider_parameters(cls, _v, values) -> ACT3MultiAgentEnvEppParameters:
        """Create the episode parameter provider for this configuration"""

        for key in ["reference_store", "dones", "simulator_reset_parameters"]:
            assert key in values.data

        reference_parameters: dict[str, Parameter | ParameterWrapper] = {
            ref_name: ref_value
            for ref_name, ref_value in values.data["reference_store"].items()
            if isinstance(ref_value, Parameter | ParameterWrapper)
        }
        world_parameters: dict[str, dict[str, Parameter | ParameterWrapper]] = {}
        for functor in values.data["dones"].world:
            functor.add_to_parameter_store(world_parameters)

        task_parameters: dict[str, dict[str, dict[str, Parameter | ParameterWrapper]]] = {}
        for agent, task_dones in values.data["dones"].task.items():
            agent_task_parameters: dict[str, dict[str, Parameter | ParameterWrapper]] = {}
            for functor in task_dones:
                functor.add_to_parameter_store(agent_task_parameters)
            task_parameters[agent] = agent_task_parameters

        shared_parameters: dict[str, dict[str, Parameter | ParameterWrapper]] = {}
        for functor in values.data["dones"].shared:
            functor.add_to_parameter_store(shared_parameters)

        sim_parameters_flat = {
            name: param
            for name, param in flatten_dict.flatten(values.data["simulator_reset_parameters"]).items()
            if isinstance(param, Parameter | ParameterWrapper)
        }
        for param_name, param_value in sim_parameters_flat.items():
            if param_name[0] != "platforms":
                continue

            for key, dep_param in param_value.config.dependent_parameters.items():
                param_value.config.dependent_parameters[key] = replace_magic_strings(dep_param, platform_name=param_name[1])

        sim_parameters = flatten_dict.unflatten(sim_parameters_flat)

        return ACT3MultiAgentEnvEppParameters(
            world=world_parameters,
            task=task_parameters,
            shared=shared_parameters,
            reference_store=reference_parameters,
            simulator_reset=sim_parameters,
        )

    @field_validator("epp_registry", mode="before")
    def construct_epp_registry_if_necessary_and_validate(cls, epp_registry, values):
        """
        validates the Episode Parameter provider registry
        """
        if epp_registry is None:
            assert (
                "episode_parameter_provider" in values.data
            ), "The validator for `episode_parameter_provider` encountered an exception and did not get passed to this validator"
            assert (
                "episode_parameter_provider_parameters" in values.data
            ), "The validator for `episode_parameter_provider_parameters` encountered an exception and did not get passed to this validator"

            epp_registry = {}
            env_epp_parameters = dict(values.data["episode_parameter_provider_parameters"])
            flat_env_epp_parameters = flatten_dict.flatten(env_epp_parameters)
            env_epp = values.data["episode_parameter_provider"].build(parameters=flat_env_epp_parameters)
            epp_registry[ACT3MultiAgentEnv.episode_parameter_provider_name] = env_epp

            for agent_id, agent_info in values.data["agents"].items():
                agent = agent_info.class_config.agent(
                    agent_name=agent_id, platform_names=agent_info.platform_names, **agent_info.class_config.config
                )
                epp_registry[agent_id] = agent.config.epp

        if ACT3MultiAgentEnv.episode_parameter_provider_name not in epp_registry:
            raise ValueError(f"Missing EPP for '{ACT3MultiAgentEnv.episode_parameter_provider_name}'")

        for agent_id in values.data["agents"]:
            if agent_id not in epp_registry:
                raise ValueError(f"Missing EPP for '{agent_id}'")

        for key, epp in epp_registry.items():
            if not isinstance(epp, EpisodeParameterProvider):
                raise TypeError(
                    f"Invalid type for epp_registry['{key}']: {type(epp)}, only {EpisodeParameterProvider.__qualname__} allowed"
                )

        return epp_registry


class ACT3MultiAgentEnv(BaseCorlMultiAgentEnv):
    """
    ACT3MultiAgentEnv create a RLLIB MultiAgentEnv environment. The following class is intended to wrap
    the interactions with RLLIB and the backend simulator environment. All items here are intended to be
    common parts for running the RLLIB environment with ${simulator} being the unique interaction parts.

    1. Includes wrapping the creation of the simulator specific to run
    2. Includes interactions with the dones, rewards, and glues
    3. etc...
    """

    def __init__(self, config: dict[str, typing.Any]) -> None:  # noqa: PLR0915
        """
        __init__ initializes the rllib multi agent environment

        Parameters
        ----------
        config : ray.rllib.env.env_context.EnvContext
            Passed in configuration for setting items up.
            Must have a 'simulator' key whose value is a BaseIntegrator type
        """
        # this copy protects the callers config from getting corrupted by anything pydantic tries to do
        # really only validators marked with `pre` can do any damage to it
        self.config: ACT3MultiAgentEnvValidator
        super().__init__(config)

        self.seed(1)
        # setup default instance variables
        self._actions: list = []
        self._obs_buffer = ObsBuffer()
        self._reward: dict = {}
        self._done: dict[str, list[DoneFuncBase]] = {}
        self._info: OrderedDict = OrderedDict()
        self._episode_length: int = 0
        self._episode: int = 0
        self._episode_id = None

        # agent glue dict is a mapping from agent id to a dict with keys for the glue names
        # and values of the actual glue object
        self._agent_glue_dict: OrderedDict = OrderedDict()
        self._agent_glue_obs_export_behavior: OrderedDict = OrderedDict()

        # Create the logger
        self._logger = logging.getLogger(ACT3MultiAgentEnv.__name__)

        # Extra simulation init args
        # assign the new output_path with the worker index back to the config for the sim/integration output_path
        extra_sim_init_args: dict[str, typing.Any] = {
            "output_path": str(self.config.output_path),
            "worker_index": self.config.worker_index,
            "vector_index": self.config.vector_index if self.config.vector_index else 0,
        }

        self._agent_dict, extra_sim_init_args["agent_configs"] = env_creation.create_agent_sim_configs(
            self.config.agents,
            self.config.agent_platforms,
            self.config.simulator.type,
            self.config.epp_registry,
            multiple_workers=(self.config.num_workers > 0),
        )
        self._trainable_agent_dict = {k: v for (k, v) in self._agent_dict.items() if v.trainable}
        self._non_trainable_agent_dict = {k: v for (k, v) in self._agent_dict.items() if not (v.trainable)}

        # generate a mapping of agents to platforms and platforms to agents
        # for quick access
        self.agent_to_platforms: dict[str, list[str]] = {}
        self.platform_to_agents: dict[str, list[str]] = {}

        for agent_name, agent_class in self._agent_dict.items():
            self.agent_to_platforms[agent_name] = []
            for platform_name in agent_class.platform_names:
                self.agent_to_platforms[agent_name].append(platform_name)
                if platform_name in self.platform_to_agents:
                    self.platform_to_agents[platform_name].append(agent_name)
                else:
                    self.platform_to_agents[platform_name] = [agent_name]

        # generate a dictionary that keeps track of which platforms that an
        # agent class controls are still operable, when all platforms are inoperable
        # the agent_class will be marked done
        self.agent_platform_inoperable_status = {}
        for agent_name, agent_platforms in self.agent_to_platforms.items():
            self.agent_platform_inoperable_status[agent_name] = {platform_name: False for platform_name in agent_platforms}

        def compute_lcm(values: list[fractions.Fraction]) -> float:
            assert len(values) > 0
            lcm = values[0].denominator
            for v in values:
                lcm = lcm // math.gcd(lcm, v.denominator) * v.denominator
            if lcm == 1:
                lcm = values[0].numerator
                for v in values:
                    lcm = math.gcd(lcm, v.numerator)
                lcm = 1 / lcm  # type: ignore
            return 1.0 / lcm

        max_rate = self.config.max_agent_rate
        self._agent_periods = {
            agent_id: fractions.Fraction(1.0 / agent.frame_rate).limit_denominator(max_rate) for agent_id, agent in self._agent_dict.items()
        }
        self._agent_process_time: dict[str, float] = defaultdict(lambda: sys.float_info.min)
        self.sim_period = compute_lcm(list(self._agent_periods.values()))
        extra_sim_init_args["frame_rate"] = 1.0 / self.sim_period

        # Debug logging
        self._logger.debug(f"output_path : {self.config.output_path}")

        # Sample parameter provider
        default_parameters = self.config.epp.config.parameters

        # Generate topological graph
        # The topological graph only guarantees order necessary to meet dependencies.  Otherwise, the order
        # is unchanged.  If the order of self.config.epp.config.parameters is not constant, the order of
        # random number evaluation can change.  On average, this should still provide good training experiences
        # for the agent; however, it can complicate repeatability studies.  Therefore, set the input of the
        # topological graph to a constant ordering by sorting the default_parameters.
        topological_graph = {}
        for param_path in sorted(default_parameters):
            param = default_parameters[param_path]
            topological_graph[param_path] = set(param.config.dependent_parameters.values())

        self.epp_param_order = tuple(TopologicalSorter(topological_graph).static_order())
        flat_variable_store: dict[tuple[str, ...], typing.Any] = {}

        for k in self.epp_param_order:
            v = default_parameters[k]
            param_value = v.get_value(self.rng, flat_variable_store)
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

        self._local_variable_store: dict[str, typing.Any] = flatten_dict.unflatten(flat_variable_store)
        for agent in self._agent_dict.values():
            agent.fill_parameters(rng=self.rng, default_parameters=True)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # DO NOT CHANGE THIS>>>> STUPID ISSUE CAUSED BY DYNAMIC RESETS OF ENV
        # - RLLIB DOES NOT ALLOW US TO RECREATE THE ENV
        # - We do not want multiple envs
        # ------------>BEGIN<------------
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if not hasattr(self, "_simulator"):
            self._simulator: BaseSimulator = self.config.simulator.build(**extra_sim_init_args)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # ------------>END<------------
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self._state, self._sim_reset_args = self._reset_simulator(extra_sim_init_args["agent_configs"])

        # Make the glue objects from the glue mapping now that we have a simulator created
        self._make_glues()

        # create dictionary to hold done history
        self.__setup_state_history()

        # Create the observation and action space now that we have the glue
        # action space operations
        self._action_prop = DictProp(
            spaces=self.__agent_aggregator(agent_list=self._trainable_agent_dict.keys(), agent_function=lambda agent, _: agent.action_prop)
        )
        self._raw_action_space = self._action_prop.create_space()
        gymnasium_space_sort(self._raw_action_space)
        self._action_units = self._action_prop.get_units()
        self.action_space = gymnasium.spaces.Dict(
            self.__agent_aggregator(
                agent_list=self._trainable_agent_dict.keys(), agent_function=lambda agent, _: agent.normalized_action_space
            )
        )
        gymnasium_space_sort(self.action_space)
        self.full_action_space = gymnasium.spaces.Dict(
            self.__agent_aggregator(agent_list=self.agent_dict.keys(), agent_function=lambda agent, _: agent.normalized_action_space)
        )
        gymnasium_space_sort(self.full_action_space)

        # obs space operations
        self._observation_prop = DictProp(
            spaces=self.__agent_aggregator(
                agent_list=self._trainable_agent_dict.keys(), agent_function=lambda agent, _: agent.observation_prop
            )
        )
        self._raw_observation_space = self._observation_prop.create_space()
        self._observation_units = self._observation_prop.get_units()
        self.observation_space = gymnasium.spaces.Dict(
            self.__agent_aggregator(
                agent_list=self._trainable_agent_dict.keys(), agent_function=lambda agent, _: agent.normalized_observation_space
            )
        )
        self.full_observation_space = gymnasium.spaces.Dict(
            self.__agent_aggregator(agent_list=self.agent_dict.keys(), agent_function=lambda agent, _: agent.normalized_observation_space)
        )

        self._set_space_seeds()

        # full obs
        self._full_observation_prop = DictProp(
            spaces=self.__agent_aggregator(agent_list=self.agent_dict.keys(), agent_function=lambda agent, _: agent.observation_prop)
        )
        self._full_observation_units = self._full_observation_prop.get_units()

        self._shared_done: DoneDict = DoneDict()
        self._done_info: OrderedDict = OrderedDict()
        self._reward_info: OrderedDict = OrderedDict()

        self._episode_init_params: dict

        self.done_string = ""
        self._agent_ids = set(self._raw_action_space.spaces.keys())

        self._skip_action = False
        self._agent_ids = list(self._agent_dict.keys())
        self._agent_ids_trainable = list(self._trainable_agent_dict.keys())
        self._agent_ids_non_trainable = list(self._non_trainable_agent_dict.keys())
        self.action_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True

    @staticmethod
    def get_validator() -> type[ACT3MultiAgentEnvValidator]:
        """Get the validator for this class."""
        return ACT3MultiAgentEnvValidator

    def reset(self, *, seed=None, options=None) -> tuple[OrderedDict, OrderedDict]:
        if seed is not None:
            self.seed(seed)
        # Sample parameter provider
        current_parameters, self._episode_id, env_epp_ctx = self.config.epp.get_params(self.rng)

        flat_variable_store: dict[tuple[str, ...], typing.Any] = {}
        for k in self.epp_param_order:
            v = current_parameters[k]
            param_value = v.get_value(self.rng, flat_variable_store)
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
        self._local_variable_store = flatten_dict.unflatten(flat_variable_store)

        # Reset all agents. This should be done prior to calling other
        # functionality on the agents in case the agents' reset()
        # changes agent attributes used to generate items like variable
        # store parameters, rewards, etc.
        for agent in self._agent_dict.values():
            agent.reset()

        for agent in self._agent_dict.values():
            agent.fill_parameters(self.rng, env_epp_ctx)

        # 3. Reset the Done and Reward dictionaries for the next iteration
        self._make_rewards()
        self._done = {}
        self._done_info.clear()
        self._make_dones()
        self._shared_done = self._make_shared_dones()

        self._reward = {}

        # generate a dictionary that keeps track of which platforms that an
        # agent class controls are still operable, when all platforms are inoperable
        # the agent_class will be marked done
        self.agent_platform_inoperable_status = {}
        for agent_name, agent_platforms in self.agent_to_platforms.items():
            self.agent_platform_inoperable_status[agent_name] = {platform_name: False for platform_name in agent_platforms}

        self._set_default_done_reward()
        # 4. Reset the simulation/integration
        self._state, self._sim_reset_args = self._reset_simulator()
        self._episode_length = 0
        self._actions.clear()
        self._episode += 1

        self._agent_process_time.clear()

        #####################################################################
        # Make glue sections - Given the state of the simulation we need to
        # update the platform interfaces.
        #####################################################################
        self._make_glues()

        #####################################################################
        # get observations
        # For each configured agent read the observations/measurements
        #####################################################################
        agent_list = list(self._agent_dict.keys())
        # The following loop guarantees that during training that the glue
        # states start with valid values for rates. The number of recommended
        # steps for sim is at least equal to the depth of the rate observation
        # tree (ex: speed - 2, acceleration - 3, jerk - 4) - recommend defaulting
        # to 4 as we do not go higher thank jerk
        # 1 step is always added for the initial obs in reset
        # 1Hz 10 Seconds = 10 --- 2Hz 10 seconds = 20 --- .5 Hz 10 Seconds = 5
        warmup_steps = math.ceil(self.config.sim_warmup_seconds * self._state.sim_update_rate_hz)
        platforms: set[str] = set()
        for _ in range(warmup_steps):
            self._state = self._simulator.step(platforms)
        self._obs_buffer.next_observation = self.__get_observations_from_glues(agent_list)
        self._obs_buffer.update_obs_pointer()
        self.__setup_state_history()

        for platform_name in self._state.sim_platforms:
            # self._state.episode_history[platform_name].clear()
            self._state.episode_state[platform_name] = OrderedDict()
        for agent_name in self._agent_dict:
            self._state.agent_episode_state[agent_name] = OrderedDict()

        self._create_actions(self._agent_dict, self._obs_buffer.observation)

        self._info.clear()
        self.__get_info_from_glue(agent_list)
        trainable_info = self._get_trainable_info(agent_list)

        training_obs = self._create_training_observations(agent_list, self._obs_buffer)

        return training_obs, trainable_info

    def _reset_simulator(self, agent_configs=None) -> tuple[BaseSimulatorState, dict[str, typing.Any]]:
        sim_reset_args = copy.deepcopy(self.config.simulator_reset_parameters)
        v_store = self._local_variable_store
        deepmerge.always_merger.merge(sim_reset_args, v_store.get("simulator_reset", {}))

        self._process_references(sim_reset_args, v_store)

        for agent_data in self._agent_dict.values():
            deepmerge.always_merger.merge(
                sim_reset_args,
                {"platforms": {platform_name: agent_data.get_simulator_reset_parameters() for platform_name in agent_data.platform_names}},
            )

        if agent_configs is not None:
            sim_reset_args["agent_configs_reset"] = agent_configs

        return self._simulator.reset(sim_reset_args), sim_reset_args

    def _process_references(self, sim_reset_args: dict, v_store: dict) -> None:
        """Process the reference store look ups for the position data

        Parameters
        ----------
        sim_reset_args : dict
            The simulator reset parameters
        v_store : dict
            The variable store
        """
        plat_str = "platforms"
        if plat_str in sim_reset_args:
            pos_str = "position"
            ref_str = "reference"
            for plat_k, plat_v in sim_reset_args[plat_str].items():
                if pos_str in plat_v:
                    for position_k, position_v in plat_v[pos_str].items():
                        if isinstance(position_v, dict) and ref_str in position_v:
                            sim_reset_args[plat_str][plat_k][pos_str][position_k] = v_store["reference_store"].get(
                                position_v[ref_str], self.config.reference_store[position_v[ref_str]]
                            )

    def _remove_non_trainable_agent_ids(self, agent_list: typing.Iterable[str]) -> list[str]:
        """
        Remove any agent IDs corresponding to non-trainable agents.
        """

        return [a for a in agent_list if a not in self._agent_ids_non_trainable]

    def _get_operable_agent_platforms(self):
        return {
            name
            for name, item in self.state.sim_platforms.items()
            if item.operable and not self.state.episode_state.get(name, {}) and self.platform_to_agents.get(name, None)
        }

    def _get_operable_agents(self):
        """Determines which agents are operable in the sim, this becomes stale after the simulation is stepped"""

        operable_agents = []
        for agent_name, platform_inoperability_status in self.agent_platform_inoperable_status.items():
            inoperability_values = platform_inoperability_status.values()

            if any(inoperability_values):
                self._agent_dict[agent_name].set_removed(platform_inoperability_status)
            if not all(inoperability_values):
                operable_agents.append(agent_name)

        return operable_agents

    def _get_trainable_info(self, agent_list: typing.Collection[str]) -> typing.OrderedDict:
        """Returns the training info"""

        agent_list = self._remove_non_trainable_agent_ids(agent_list)

        trainable_info = get_dictionary_subset(self._info, agent_list)

        # add platform obs and env data to trainable_info (for use by custom policies)
        for agent_id in agent_list:
            if agent_id not in trainable_info:
                trainable_info[agent_id] = {}

            trainable_info[agent_id]["env"] = {"sim_period": self.sim_period, "sim_time": self._state.sim_time}

            platform_indices = {platform_id: idx for idx, platform_id in enumerate(self.platform_to_agents)}
            agents_to_platform_indices = {
                agent: [platform_indices[platform] for platform in platforms] for agent, platforms in self.agent_to_platforms.items()
            }

            trainable_info[agent_id]["platform_to_index"] = platform_indices
            trainable_info[agent_id]["agents_to_platform_indices"] = agents_to_platform_indices

        return trainable_info

    def step(self, action_dict: dict):  # noqa: PLR0915,PLR0912
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

            obs (StateDict): New observations for each ready agent.
                episode is just started, the value will be None.
            dones (StateDict): Done values for each ready agent. The
                special key "__all__" (required) is used to indicate env
                termination.
            infos (StateDict): Optional info values for each agent id.
        """
        self._episode_length += 1

        operable_platforms = self._get_operable_agent_platforms()
        operable_agents = self._get_operable_agents()

        for agent_id in self._non_trainable_agent_dict:
            _observation = self._obs_buffer.observation.get(agent_id)
            if _observation is None:
                continue
            _action = self._non_trainable_agent_dict[agent_id].create_action(observation=_observation)  # type: ignore
            action_dict[agent_id] = _action

        # look to add this bugsplat, but this check won't work for multi fps
        # if set(operable_agents.keys()) != set(action_dict.keys()):
        #     raise RuntimeError("Operable_agents and action_dict keys differ!"
        #                        f"operable={set(operable_agents.keys())} != act={set(action_dict.keys())} "
        #                        "If this happens that means either your platform is not setting non operable correctly"
        #                        " (if extra keys are in operable set) or you do not have a done condition covering "
        #                        "a condition where your platform is going non operable. (if extra keys in act)")
        raw_action_dict = {} if self._skip_action else self._apply_action(operable_agents, action_dict)

        # Save current action for future debugging
        if not self.config.disable_action_history:
            self._actions.append(action_dict)

        platforms_to_process = set()
        for agent_id in action_dict:
            for platform_name in self.agent_to_platforms[agent_id]:
                if platform_name in operable_platforms:
                    platforms_to_process.add(platform_name)

        try:
            self._state = self._simulator.step(platforms_to_process)
        except ValueError as err:
            if self.config.save_state_pickle:
                self._save_state_pickle(err)

        # MTB - Changing to not replace operable_agents variable
        #       We calculate observations on agents operable after sim step
        #       - This is done because otherwise observations would be invalid
        #       Calculate Dones/Rewards on agents operable before sim step
        #       - This is done because if an agent "dies" it needs to have a final done calculated
        operable_platforms_after_step = self._get_operable_agent_platforms()
        # get the difference in operable platforms to find those that are inoperable
        # then we mark it inoperable so that _get_operable_agents can correctly do it's job
        new_inoperable_platforms = operable_platforms - operable_platforms_after_step
        for platform_name in new_inoperable_platforms:
            for agent_name in self.platform_to_agents[platform_name]:
                self.agent_platform_inoperable_status[agent_name][platform_name] = True

        operable_agents_after_step = self._get_operable_agents()

        #####################################################################
        # get next observations - For each configured platform read the
        # observations/measurements
        #####################################################################
        self._obs_buffer.next_observation = self.__get_observations_from_glues(operable_agents_after_step)

        self._info.clear()
        self.__get_info_from_glue(operable_agents_after_step)

        #####################################################################
        # Process the done conditions
        # 1. Reset the rewards from the last step
        # 2. loops over all agents and processes the reward conditions per
        #    agent
        #####################################################################

        platforms_done = self.__get_done_from_platforms(operable_platforms, raw_action_dict=raw_action_dict)

        expected_done_keys = operable_platforms
        if set(platforms_done.keys()) != expected_done_keys:
            raise RuntimeError(
                f'Local dones do not match expected keys.  Received "{platforms_done.keys()}".  Expected "{expected_done_keys}".'
            )

        shared_dones, shared_done_info = self._shared_done(
            observation=self._obs_buffer.observation,
            action=raw_action_dict,
            next_observation=self._obs_buffer.next_observation,
            next_state=self._state,
            observation_space=self._raw_observation_space,
            observation_units=self._observation_units,
            local_dones=copy.deepcopy(platforms_done),
            local_done_info=copy.deepcopy(self._done_info),
        )

        if shared_dones.keys():
            if set(shared_dones.keys()) != expected_done_keys:
                raise RuntimeError(
                    f'Shared dones do not match expected keys.  Received "{shared_dones.keys()}".  Expected "{expected_done_keys}".'
                )
            for key in expected_done_keys:
                platforms_done[key] |= shared_dones[key]

            assert shared_done_info is not None

            local_done_info_keys = set(self._done_info.keys())
            shared_done_info_keys = set(shared_done_info)
            common_keys = local_done_info_keys & shared_done_info_keys
            if not common_keys:
                raise RuntimeError(f'Dones do not have common keys: "{common_keys}"')

            for platform_name in filter(lambda x: x in shared_done_info, self._done_info):
                self._done_info[platform_name].update(shared_done_info[platform_name])

        # we now have all platforms that are done
        # convert these platform done's to agent dones
        # update state to map platform episode state to agent episode state
        for platform_name, _ in filter(lambda x: x[1], platforms_done.items()):
            for agent_name in self.platform_to_agents[platform_name]:
                self.agent_platform_inoperable_status[agent_name][platform_name] = True
                for done_name, done_status in self._state.episode_state[platform_name].items():
                    self._state.agent_episode_state[agent_name][done_name] = done_status

        # compute agent dones from the platform dones
        # update episode state to hold information about agent done status
        agents_done = {}
        for agent_name, agent_platform_dones in self.agent_platform_inoperable_status.items():
            agents_done[agent_name] = all(agent_platform_dones.values())

        # calculate if all agents are done
        if self.config.end_episode_on_first_agent_done:
            agents_done["__all__"] = any(agents_done.values())
        else:
            agents_done["__all__"] = all(agents_done.values())

        # Update agent_episode_state based on if shared dones modified episode_state after an agent was done
        if agents_done["__all__"]:
            for platform_name in self.state.episode_state:
                for agent_name in self.platform_to_agents.get(platform_name, []):
                    for done_name, done_status in self._state.episode_state[platform_name].items():
                        self._state.agent_episode_state[agent_name][done_name] = done_status

        # Tell the simulator to mark the episode complete
        if agents_done["__all__"]:
            self._simulator.mark_episode_done(self._done_info, self._state.episode_state, self.config.metadata)

        self._reward.clear()

        if agents_done["__all__"]:
            agents_to_process_this_timestep = operable_agents
        else:

            def do_process_agent(self, agent_id) -> bool:
                frame_rate = self._agent_periods[agent_id].numerator / self._agent_periods[agent_id].denominator
                return self._state.sim_time >= self._agent_process_time[agent_id] + frame_rate - self.config.timestep_epsilon

            agents_to_process_this_timestep = list(filter(partial(do_process_agent, self), operable_agents))

        for agent_id in agents_to_process_this_timestep:
            self._agent_process_time[agent_id] = self._state.sim_time

        reward = self.__get_reward_from_agents(agents_to_process_this_timestep, raw_action_dict=raw_action_dict)

        self._simulator.save_episode_information(
            self.done_info, self.reward_info, self._obs_buffer.next_observation, self._full_observation_units
        )
        # copy over observation from next to previous - There is no real reason to deep
        # copy here. The process of getting a new observation from the glue copies. All
        # we need to do is maintain the order of two buffers!!!.
        # Tested with: They are different and decreasing as expected
        #   print(f"C: {self._obs_buffer.observation['blue0']['ObserveSensor_Sensor_Fuel']}")
        #   print(f"N: {self._obs_buffer.next_observation['blue0']['ObserveSensor_Sensor_Fuel']}")
        self._obs_buffer.update_obs_pointer()

        # Note that `_create_training_observations` and `_get_trainable_info`
        # contain a filter for trainable agents.
        trainable_observations = self._create_training_observations(agents_to_process_this_timestep, self._obs_buffer)
        trainable_info = self._get_trainable_info(agents_to_process_this_timestep)

        # Filters 'agent IDs to process this timestep' so that it only
        # includes IDs of trainable agents.
        trainable_agents_to_process_this_timestep = self._remove_non_trainable_agent_ids(agents_to_process_this_timestep)
        trainable_rewards = get_dictionary_subset(reward, trainable_agents_to_process_this_timestep)
        trainable_dones = get_dictionary_subset(agents_done, [*list(trainable_agents_to_process_this_timestep), "__all__"])

        # if not done all, delete any platforms from simulation that are done, so they don't interfere
        # go through done's and update inoperable platform list so that operable platforms can be updated
        if not agents_done["__all__"]:
            platforms_deleted = set()
            for platform_name, platform_done in platforms_done.items():
                if platform_done:
                    self.simulator.delete_platform(platform_name)
                    platforms_deleted.add(platform_name)

        #####################################################################
        # return results to RLLIB - Note that RLLIB does not do a recursive
        # isinstance call and as such need to make sure items are
        # OrderedDicts
        #####################################################################
        return trainable_observations, trainable_rewards, trainable_dones, {"__all__": trainable_dones["__all__"]}, trainable_info

    def __get_done_from_platforms(self, alive_platforms: typing.Iterable[str], raw_action_dict):
        def or_merge(config, path, base, nxt):
            return base or nxt

        merge_strategies = copy.deepcopy(deepmerge.DEFAULT_TYPE_SPECIFIC_MERGE_STRATEGIES)
        merge_strategies.append((bool, or_merge))
        or_merger = deepmerge.Merger(merge_strategies, [], [])

        done = OrderedDict()

        for platform_name in filter(lambda x: x in self._done, alive_platforms):
            platform_is_done = False
            done_info = {}
            for platform_done in self._done[platform_name]:
                ret = platform_done(
                    observation=self._obs_buffer.observation,
                    action=raw_action_dict,
                    next_observation=self._obs_buffer.next_observation,
                    next_state=self._state,
                    observation_space=self._raw_observation_space,
                    observation_units=self._observation_units,
                )
                platform_is_done = platform_is_done or ret
                done_info[platform_done.name] = ret

            done[platform_name] = platform_is_done
            # get around reduction
            or_merger.merge(self._done_info.setdefault(platform_name, {}), done_info)

            self._done_info[platform_name] = done_info

        return done

    def __get_reward_from_agents(self, alive_agents: typing.Iterable[str], raw_action_dict):
        reward = OrderedDict()
        for agent_id in alive_agents:
            agent_class = self._agent_dict[agent_id]
            agent_reward, reward_info = agent_class.get_rewards(
                observation=self._obs_buffer.observation,
                action=raw_action_dict,
                next_observation=self._obs_buffer.next_observation,
                state=self._state,
                next_state=self._state,
                observation_space=self._raw_observation_space,
                observation_units=self._observation_units,
            )
            # it is possible to have a HL policy that does not compute an reward
            # in this case just return a zero for reward value
            reward[agent_id] = agent_reward
            self._reward_info[agent_id] = reward_info
        return reward

    def _set_default_done_reward(self):
        """
        Populate the done/rewards with default values
        """
        for key in self._agent_dict:
            self._reward[key] = 0
            self._shared_done[key] = False
        self._shared_done[SharedDoneFuncBase.ALL] = False

    def _create_training_observations(self, alive_agents: typing.Iterable[str], observations: ObsBuffer) -> OrderedDict:
        """
        Filters and normalizes observations (the sample of the space) using the glue normalize functions.

        Parameters
        ----------
        alive_agents:
            The agents that are still alive
        observations:
            The observations

        Returns
        -------
        OrderedDict:
            the filtered/normalized observation samples
        """

        alive_agents = self._remove_non_trainable_agent_ids(alive_agents)

        this_steps_obs = OrderedDict()
        for agent_id in alive_agents:
            if agent_id in observations.observation:
                this_steps_obs[agent_id] = observations.observation[agent_id]
            elif agent_id in observations.next_observation:
                this_steps_obs[agent_id] = observations.next_observation[agent_id]
            else:
                raise RuntimeError(
                    "ERROR: _create_training_observations tried to retrieve obs for this training step"
                    f" but {agent_id=} was not able to be found in either the current obs data or the "
                    " obs from the previous timestep as a fallback"
                )

        # sanity check will fail on quantities, and we definitely don't want to send quantities into an RL
        # framework
        unit_extracted_obs = tree.map_structure(lambda v: v.m if isinstance(v, Quantity) else v, this_steps_obs)
        # Sanity checks and Scale - ensure run first time and run only every N times...
        # Same as RLLIB - This can add a bit of time as we are exploring complex dictionaries
        # default to every time if not specified... Once the limits are good we it is
        # recommended to increase this for training
        if self.config.deep_sanity_check and self._episode_length % self.config.sanity_check_obs == 0:
            try:
                self._sanity_check(self._raw_observation_space, unit_extracted_obs)
            except ValueError as err:
                if self.config.save_state_pickle:
                    self._save_state_pickle(err)

        return OrderedDict(
            self.__agent_aggregator(
                agent_list=alive_agents,
                agent_function=lambda agent, agent_id, all_agent_obs: agent.create_training_observations(all_agent_obs[agent_id]),
                all_agent_obs=unit_extracted_obs,
            )
        )

    def __get_observations_from_glues(self, alive_agents: typing.Iterable[str]) -> OrderedDict:
        """
        Gets the observation dict from all the glue objects for each agent

        Returns
        -------
        OrderedDict:
            The observation dict from all the glues
        """
        return_observation: OrderedDict = OrderedDict()
        for agent_id in alive_agents:
            agent_class = self._agent_dict[agent_id]

            glue_obj_obs = agent_class.get_observations()
            if len(glue_obj_obs) > 0:
                return_observation[agent_id] = glue_obj_obs
        return return_observation

    def _apply_action(self, operable_agents, action_dict):
        raw_action_dict = OrderedDict()
        for agent_id in operable_agents:
            agent_class = self._agent_dict[agent_id]
            if agent_id in action_dict:
                raw_action_dict[agent_id] = agent_class.apply_action(action_dict[agent_id], self._obs_buffer.observation[agent_id])
        return raw_action_dict

    def __get_info_from_glue(self, alive_agents: typing.Iterable[str]):
        for agent_id in alive_agents:
            agent_class = self._agent_dict[agent_id]
            glue_obj_info = agent_class.get_info_dict()
            if len(glue_obj_info) > 0:
                self._info[agent_id] = glue_obj_info

    def _make_glues(self) -> None:
        """ """
        env_ref_stores = [self._local_variable_store.get("reference_store", {}), self.config.reference_store]
        for agent, agent_class in self._agent_dict.items():
            agent_plats = {platform_name: self._get_platform_by_name(platform_name) for platform_name in agent_class.platform_names}
            agent_class.make_glues(agent_plats, agent, env_ref_stores=env_ref_stores)

    def _make_rewards(self) -> None:
        """ """
        env_ref_stores = [self._local_variable_store.get("reference_store", {}), self.config.reference_store]

        for agent, agent_class in self._agent_dict.items():
            agent_class.make_rewards(agent, env_ref_stores=env_ref_stores)

    def _make_dones(self) -> None:
        """
        register done functors to platforms, but done conditions are assigned to a specific platform
        """
        env_ref_stores = [self._local_variable_store.get("reference_store", {}), self.config.reference_store]
        # 1Hz 10 Seconds = 10 --- 2Hz 10 seconds = 20 --- .5 Hz 10 Seconds = 5
        warmup_steps = math.ceil(self.config.sim_warmup_seconds * self._state.sim_update_rate_hz)
        episode_length_done = Functor(
            functor=EpisodeLengthDone,
            config={"horizon": corl_quantity()(value=(self.config.horizon + warmup_steps) * self.sim_period, units="sec")},
        )

        platform_dones_collection: dict[str, list[DoneFuncBase]] = {}
        platform_done_dicts = {}
        for agent_class in self._agent_dict.values():
            env_params = {
                platform_name: [
                    self._local_variable_store.get("world", {}),
                    self._local_variable_store.get("task", {}).get(platform_name, {}),
                ]
                for platform_name in agent_class.platform_names
            }
            tmp = agent_class.make_platform_dones(env_params=env_params, env_ref_stores=env_ref_stores)
            for platform_name, platform_dones in tmp.items():
                if platform_name in platform_dones_collection:
                    platform_dones_collection[platform_name].extend(platform_dones)
                else:
                    platform_dones_collection[platform_name] = platform_dones

        for platform_name in self.platform_to_agents:
            platform_dones = platform_dones_collection.get(platform_name, [])
            for done_dict in chain(self.config.dones.world, self.config.dones.task[platform_name], [episode_length_done]):
                platform_dones.append(
                    done_dict.create_functor_object(
                        param_sources=[
                            self._local_variable_store.get("world", {}),
                            self._local_variable_store.get("task", {}).get(platform_name, {}),
                        ],
                        ref_sources=env_ref_stores,
                        platform_name=platform_name,
                    )
                )
            # platform_done_dict = DoneDict(processing_funcs=platform_dones)  # type: ignore
            platform_done_dicts[platform_name] = platform_dones

        self._done = platform_done_dicts

    def _make_shared_dones(self) -> DoneDict:
        """
        _get_shared_done_functors gets and initializes the
        shared done dict used for this iteration

        The shared done dictionary does not correspond to individual
        agents but looks sharedly at all agents.

        this will be called after any updates to the simulator
        configuration during reset

        Returns
        -------
        DoneDict
            The DoneDict with functors used for this iteration
        """
        done_conditions = []
        ref_sources = [self._local_variable_store.get("reference_store", {}), self.config.reference_store]
        param_sources = [self._local_variable_store.get("shared", {})]
        for done_functor in self.config.dones.shared:
            tmp = done_functor.create_functor_object(param_sources=param_sources, ref_sources=ref_sources)
            done_conditions.append(tmp)
        return DoneDict(processing_funcs=done_conditions)

    @staticmethod
    def _create_actions(
        agents,
        observations: dict | None = None,
        rewards: dict | None = None,
        dones: dict | None = None,
        info: dict | None = None,
    ):
        if info is None:
            info = dict()
        if dones is None:
            dones = dict()
        if rewards is None:
            rewards = dict()
        if observations is None:
            observations = dict()
        for agent_name, agent_class in agents.items():
            agent_class.create_next_action(
                observations.get(agent_name), rewards.get(agent_name), dones.get(agent_name), info.get(agent_name)
            )

    def __agent_aggregator(self, agent_list, agent_function: typing.Callable, **kwargs) -> dict:
        """
        __agent_aggregator calls function on all agents and places the output in the in a dictionary.
        Dictionary key set to the agent name and values are set to the return value of the method

        Parameters
        ----------
        agent_function
            A function that takes an agent: BaseAgent as an argument
        Returns
        -------
            A dictionary of agent names to method output
        """
        # init our return Dict
        agg_dict = {}
        # loop over all the agents and their glue_name_obj_pairs list
        for agent_id in agent_list:
            agent_class = self._agent_dict[agent_id]
            agg_dict[agent_id] = agent_function(agent_class, agent_id, **kwargs)
            # if this agent provided anything add it to the return space
        return agg_dict

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
    def observation(self) -> OrderedDict:
        """
        observation get the observation for the agents in this environment

        Returns
        -------
        OrderedDict
            the dict holding the observations for the agents
        """
        return self._obs_buffer.observation

    @property
    def episode_id(self) -> int | None:
        """
        get the current episode parameter provider episode id

        Returns
        -------
        int or None
            the episode id
        """
        return self._episode_id

    @property
    def local_variable_store(self) -> dict[str, typing.Any]:
        """
        get the local variable store

        Returns
        -------
        Dict
            the local variable store episode id
        """
        return self._local_variable_store

    @property
    def epp(self) -> EpisodeParameterProvider:
        """
        get the Episode Parameter Provider

        Returns
        -------
        EpisodeParameterProvider
        """
        return self.config.epp

    @property
    def agent_dict(self) -> dict[str, typing.Any]:
        """
        get the full agent dict

        Returns
        -------
        dict[str, typing.Any]
            A dictionary of {agent_id, agent} for agents in env.
        """
        return self._agent_dict

    def _get_platform_by_name(self, platform_id: str) -> BasePlatform:
        platform: BasePlatform = self._state.sim_platforms.get(platform_id, None)

        if platform is None or not issubclass(platform.__class__, BasePlatform):
            self._logger.error("-" * 100)
            self._logger.error(f"{self._state}")
            for i in self._state.sim_platforms:
                self._logger.error(f"{i}")
            raise ValueError(f"{self.__class__.__name__} glue could not find a platform named {platform_id} of class BasePlatform")

        return platform

    def __setup_state_history(self):
        # self._state.episode_history = defaultdict(partial(deque, maxlen=self.config.horizon))
        pass

    def post_process_trajectory(self, agent_id, batch, episode, policy):
        """easy accessor for calling post process trajectory
            correctly

        Arguments:
            agent_id: agent id
            batch: post processed Batch - be careful modifying
        """
        self._agent_dict[agent_id].post_process_trajectory(
            agent_id, episode.worker.env._state, batch, episode, policy, episode.worker.env._reward_info  # noqa: SLF001  # noqa: SLF001
        )

    @staticmethod
    def _sanity_check(space: gymnasium.spaces.Space, space_sample: EnvSpaceUtil.sample_type) -> None:
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

    def _save_state_pickle(self, err: ValueError):
        """saves state for later debug

        Arguments:
            err {ValueError} -- Traceback for the error

        Raises:
            err: Customized error message to raise for the exception

        """
        out_pickle = str(Path(self.config.output_path) / f"sanity_check_failure_{self._episode}.pkl")
        p_dict: dict[str, typing.Any] = {}
        p_dict["err"] = str(err)

        class NumpyArrayEncoder(JSONEncoder):
            """Encode the numpy types for json"""

            def default(self, o):
                val = None
                if isinstance(
                    o,
                    np.int_ | np.intc | np.intp | np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64,
                ):
                    val = int(o)
                elif isinstance(o, np.float_ | np.float16 | np.float32 | np.float64):
                    val = float(o)
                elif isinstance(o, np.ndarray):  # This is the fix
                    val = o.tolist()
                elif isinstance(o, np.bool_):
                    val = "True" if o is True else "False"
                else:
                    val = json.JSONEncoder.default(self, o)

                return val

        def to_dict(input_ordered_dict):
            return loads(dumps(input_ordered_dict, cls=NumpyArrayEncoder))

        p_dict["action"] = self._actions
        p_dict["observation"] = self._obs_buffer.observation
        p_dict["dones"] = to_dict(self._done_info)
        # p_dict["env_config"] = copy.deepcopy(self.env_config)  # type: ignore
        p_dict["step"] = str(self._episode_length)

        with open(out_pickle, "wb") as f:
            pickle.dump(p_dict, f)

        raise ValueError(f"Error occurred: {err} \n Saving sanity check failure output pickle to file: {out_pickle}")

    def seed(self, seed=None):
        """generates environment seed through rllib

        Keyword Arguments:
            seed {[int]} -- seed to set environment with (default: {None})

        Returns:
            [int] -- [seed value]
        """
        self.rng, self.seed_value = gymnasium.utils.seeding.np_random(seed)

        # Call to self._set_space_seeds() will fail during __init__ because these attributes do not exist yet.
        with contextlib.suppress(AttributeError):
            self._set_space_seeds()

        return [self.seed_value]

    def _set_space_seeds(self) -> None:
        self.observation_space.seed(self.seed_value)
        self._raw_observation_space.seed(self.seed_value)
        self.action_space.seed(self.seed_value)
        self._raw_action_space.seed(self.seed_value)
