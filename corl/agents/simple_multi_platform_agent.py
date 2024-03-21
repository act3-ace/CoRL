"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
SimpleMultiPlatform Agent Class
"""
import collections
import typing
from functools import cached_property
from graphlib import TopologicalSorter

import gymnasium
from ray.rllib.utils.spaces.repeated import Repeated
from tree import map_structure

from corl.agents.base_agent import BaseAgent, BaseAgentParser
from corl.glues.base_glue import BaseAgentGlue, TrainingExportBehavior
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.property import DictProp, RepeatedProp
from corl.libraries.space_transformations import SpaceTransformationBase
from corl.libraries.units import corl_get_ureg


class SimpleMultiPlatformParser(BaseAgentParser):
    """
    SimpleMultiPlatformParser is currently the same as BaseAgentParser
    """


class SimpleMultiPlatform(BaseAgent):
    """
    SimpleMultiPlatform is a platform that uses the same configuration as BaseAgent
    but will control multiple platforms with the same parts, glues, rewards, and dones

    This agent does not implement behaviors for glues that do not associate with a platform
    """

    def __init__(self, **kwargs) -> None:
        self.config: SimpleMultiPlatformParser = self.get_validator()(**kwargs)
        self.agent_glue_dict: dict[str, dict[str, BaseAgentGlue]] = {}  # type: ignore
        self.agent_reward_list: list = []
        self.agent_glue_order: dict[str, tuple] = {}  # type: ignore

        self._action_space_transformation: SpaceTransformationBase | None = None
        self._obs_space_transformation: SpaceTransformationBase | None = None

        # Sample parameter provider
        # This RNG only used here.  Normal use uses the one from the environment.
        rng, _ = gymnasium.utils.seeding.np_random(0)
        self.fill_parameters(rng=rng, default_parameters=True)
        self.inoperable_platforms = {platform_name: False for platform_name in self.config.platform_names}

        self.observation_space: gymnasium.spaces.Dict
        self.action_space: gymnasium.spaces.Dict

    @staticmethod
    def get_validator() -> type[SimpleMultiPlatformParser]:
        return SimpleMultiPlatformParser

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
        self.agent_glue_dict.clear()
        for config_glues, platform_name in zip(self.config.glues, self.config.platform_names):
            platform = platforms[platform_name]
            platform_glues = {}
            sort_order = {}
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

                if glue_name in platform_glues:
                    raise RuntimeError(f"The {glue_name} glue has a unique name, but it already exists")
                # find dependent nodes for the topological search
                # this will get overwritten each platform loop, but it doesn't matter for this agent class
                dependent_glues = {extractor.fields[0] for extractor in created_glue.config.extractors.values()}
                sort_order[glue_name] = dependent_glues

                # add the glue to the agent glue dict
                # self._agent_glue_obs_export_behavior[glue_name] = glue.training_obs_behavior
                platform_glues[glue_name] = created_glue

            if len(self.config.platform_names) == 1:
                self.agent_glue_dict = platform_glues
            else:
                self.agent_glue_dict[platform_name] = platform_glues

            self.agent_glue_order[platform_name] = tuple(TopologicalSorter(sort_order).static_order())

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
        if space_getter is None:

            def default_getter(glue_obj):
                return glue_obj.observation_space

            space_getter = default_getter

        return_space = gymnasium.spaces.dict.Dict()
        # loop over all glue name and  glue_obj pairs
        for platform_name, glue_dict in self.agent_glue_dict.items():
            platform_space = gymnasium.spaces.dict.Dict()
            for glue_name in self.agent_glue_order[platform_name]:
                # call our space getter to pick which space we want,
                # for example: action_space, observation_space, normalized_action_space, normalized_observation_space
                space_def = space_getter(glue_dict[glue_name])
                # if the space is None don't add it
                if space_def:
                    platform_space[glue_name] = space_def
            if len(platform_space) > 0:
                return_space[platform_name] = platform_space
        return return_space if len(return_space) > 0 else None

    def _create_space_dict(self, space_getter: typing.Callable | None = None) -> dict[str, dict[str, DictProp]]:
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
        if space_getter is None:

            def default_getter(glue_obj):
                return glue_obj.observation_space

            space_getter = default_getter

        return_space = {}
        # loop over all glue name and  glue_obj pairs
        for platform_name, glue_dict in self.agent_glue_dict.items():
            platform_space = {}
            for glue_name in self.agent_glue_order[platform_name]:
                # call our space getter to pick which space we want,
                # for example: action_space, observation_space, normalized_action_space, normalized_observation_space
                space_def = space_getter(glue_dict[glue_name])
                # if the space is None don't add it
                if space_def:
                    platform_space[glue_name] = space_def
            if len(platform_space) > 0:
                return_space[platform_name] = platform_space
        if len(return_space) > 0:
            return return_space

        raise RuntimeError("Failed to create space dict")

    def get_observations(self) -> collections.OrderedDict:
        """
        Gets combined observation from agent glues.

        Returns
        -------
        OrderedDict
            A dictionary of glue observations in the form {glue_name: glue_observation}
        """
        return_observation: collections.OrderedDict = collections.OrderedDict()
        for platform_name, glue_dict in self.agent_glue_dict.items():
            if self.inoperable_platforms[platform_name] and glue_dict:
                return_observation[platform_name] = []
                continue
            platform_obs: collections.OrderedDict = collections.OrderedDict()
            for glue_name in self.agent_glue_order[platform_name]:
                glue_object = glue_dict[glue_name]
                if glue_obs := glue_object.get_observation(
                    platform_obs,
                    self.observation_space[platform_name],
                    self.observation_units[platform_name],
                ):
                    platform_obs[glue_name] = glue_obs
                    if glue_object.config.clip_to_space and glue_object.observation_space:
                        platform_obs[glue_name] = EnvSpaceUtil.clip_space_sample_to_space(glue_obs, glue_object.observation_space)
            if len(platform_obs) > 0:
                return_observation[platform_name] = [platform_obs]
        return return_observation

    def create_training_observations(self, observations: collections.OrderedDict):
        """
        Base class representing a trainable agent in an environment.
        """

        training_obs: dict[str, list[dict[str, typing.Any]]] = collections.OrderedDict()
        for platform_name, obs_list in observations.items():
            if len(obs_list) == 0:
                training_obs[platform_name] = []
                continue
            obs_dict = obs_list[0]
            platform_obs = collections.OrderedDict()
            for obs_name, obs in obs_dict.items():
                glue_obj = self.agent_glue_dict[platform_name][obs_name]
                if glue_obj is not None and glue_obj.config.training_export_behavior == TrainingExportBehavior.INCLUDE:
                    platform_obs[obs_name] = glue_obj.normalize_observation(obs)
            training_obs[platform_name] = [platform_obs]

        if self._obs_space_transformation:
            return self._obs_space_transformation.convert_sample(sample=training_obs)
        return training_obs

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

        raw_action_dict: dict[str, dict[str, typing.Any]] = collections.OrderedDict()
        for platform_name, platform_action in action.items():
            raw_action_dict[platform_name] = {}
            for glue_name, normalized_action in platform_action.items():
                glue_object = self.agent_glue_dict[platform_name][glue_name]
                raw_action = glue_object.unnormalize_action(normalized_action)
                raw_action = map_structure(
                    lambda x, y: corl_get_ureg().Quantity(x, y), raw_action, self.action_units[platform_name][glue_name]
                )
                raw_action_dict[platform_name][glue_name] = raw_action
                if not self.inoperable_platforms[platform_name]:
                    glue_object.apply_action(
                        raw_action,
                        last_obs,
                        self.action_space[platform_name],
                        self.observation_space[platform_name],
                        self.observation_units[platform_name],
                    )
        return raw_action_dict

    def get_info_dict(self) -> collections.OrderedDict:
        """
        Gets combined observation from agent glues.

        Returns
        -------
        OrderedDict
            A dictionary of glue observations in the form {glue_name: glue_observation}
        """
        return_info: collections.OrderedDict = collections.OrderedDict()
        for platform_name, platform_glues in self.agent_glue_dict.items():
            return_info[platform_name] = {}
            for glue_name in self.agent_glue_order[platform_name]:
                if glue_info := platform_glues[glue_name].get_info_dict():
                    return_info[platform_name][glue_name] = glue_info
        return return_info

    def set_removed(self, removed_state: dict[str, bool]):
        """
        Set the agent_removed flag for the agent's glues.

        Parameters
        ----------
        - removed_state (typing.Dict[str, bool]): The the inoperable status of the platform an agent class is controlling.
            True: the platform is no longer operable
            False: the platform is still operable

        Returns
        -------
        None
        """
        for platform, platform_glue_dict in filter(lambda x: removed_state.get(x[0], False), self.agent_glue_dict.items()):
            self.inoperable_platforms[platform] = True
            for glue in platform_glue_dict.values():
                glue.set_agent_removed(True)

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

    @cached_property
    def _observation_space(self):
        tmp = self._create_space(space_getter=lambda glue_obj: glue_obj.observation_space)
        return gymnasium.spaces.Dict(
            {platform_name: Repeated(platform_space, max_len=1) for platform_name, platform_space in tmp.spaces.items()}
        )

    @cached_property
    def action_prop(self):
        """
        Returns the action prop that defines this agents actions
        """
        tmp = self._create_space_dict(space_getter=lambda glue_obj: glue_obj.action_prop)
        assert tmp is not None
        return DictProp(spaces={platform_name: DictProp(spaces=tmp[platform_name]) for platform_name in self.agent_glue_dict})

    @cached_property
    def observation_prop(self):
        """
        Returns the action prop that defines this agents actions
        """
        tmp = self._create_space_dict(space_getter=lambda glue_obj: glue_obj.observation_prop)
        assert tmp is not None
        return DictProp(
            spaces={platform_name: RepeatedProp(child_space=tmp[platform_name], max_len=1) for platform_name in self.agent_glue_dict}
        )

    @cached_property
    def normalized_observation_space(self):
        """
        Returns agents normalized observation space
        """
        tmp = self._create_space(
            space_getter=lambda glue_obj: glue_obj.normalized_observation_space
            if glue_obj.config.training_export_behavior == TrainingExportBehavior.INCLUDE
            else None
        )
        tmp = gymnasium.spaces.Dict(
            {platform_name: Repeated(platform_space, max_len=1) for platform_name, platform_space in tmp.spaces.items()}
        )
        if self.config.policy_observation_transformation:
            self._obs_space_transformation = self.config.policy_observation_transformation.transformation(
                tmp, self.config.policy_observation_transformation.config
            )
            if self._obs_space_transformation:
                return self._obs_space_transformation.output_space
        return tmp
