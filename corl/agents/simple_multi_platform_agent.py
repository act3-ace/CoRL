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
from functools import lru_cache
from graphlib import TopologicalSorter

import gym
from ray.rllib.utils.spaces.repeated import Repeated

from corl.agents.base_agent import BaseAgent, BaseAgentParser
from corl.glues.base_glue import BaseAgentGlue, TrainingExportBehavior
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.environment_dict import RewardDict
from corl.libraries.space_transformations import SpaceTransformationBase


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

    def __init__(self, **kwargs) -> None:  # pylint: disable=W0231
        self.config: SimpleMultiPlatformParser = self.get_validator(**kwargs)
        self.agent_glue_dict: typing.Dict[str, typing.Dict[str, BaseAgentGlue]] = {}  # type: ignore
        self.agent_reward_dict = RewardDict()
        self.agent_glue_order: typing.Tuple = ()

        self._action_space_transformation: typing.Optional[SpaceTransformationBase] = None
        self._obs_space_transformation: typing.Optional[SpaceTransformationBase] = None

        # Sample parameter provider
        # This RNG only used here.  Normal use uses the one from the environment.
        rng, _ = gym.utils.seeding.np_random(0)
        self.fill_parameters(rng=rng, default_parameters=True)
        self.inoperable_platforms = {platform_name: False for platform_name in self.config.platform_names}

    @property
    def get_validator(self):
        return SimpleMultiPlatformParser

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
        self.agent_glue_dict.clear()
        for platform_name, platform in platforms.items():
            self.agent_glue_dict[platform_name] = {}
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
                    if glue_name in self.agent_glue_dict[platform_name]:
                        raise RuntimeError(f"The {glue_name} glue has a unique name, but it already exists")
                else:
                    raise RuntimeError(f"No glue name for {created_glue}")

                # find dependent nodes for the topological search
                # this will get overwritten each platform loop, but it doesn't matter for this agent class
                dependent_glues = {extractor.fields[0] for extractor in created_glue.config.extractors.values()}
                sort_order[glue_name] = dependent_glues

                self.agent_glue_dict[platform_name][glue_name] = created_glue
        self.agent_glue_order = tuple(TopologicalSorter(sort_order).static_order())

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
        for platform_name, glue_dict in self.agent_glue_dict.items():
            platform_space = gym.spaces.dict.Dict()
            for glue_name in self.agent_glue_order:

                # call our space getter to pick which space we want,
                # for example: action_space, observation_space, normalized_action_space, normalized_observation_space
                space_def = space_getter(glue_dict[glue_name])
                # if the space is None don't add it
                if space_def:
                    platform_space.spaces[glue_name] = space_def
            return_space.spaces[platform_name] = platform_space
        return return_space if len(return_space.spaces) > 0 else None

    def get_observations(self):
        """
        Gets combined observation from agent glues.

        Returns
        -------
        OrderedDict
            A dictionary of glue observations in the form {glue_name: glue_observation}
        """
        return_observation: collections.OrderedDict = collections.OrderedDict()
        for platform_name, glue_dict in self.agent_glue_dict.items():
            if self.inoperable_platforms[platform_name]:
                return_observation[platform_name] = []
                continue
            platform_obs = collections.OrderedDict()
            for glue_name in self.agent_glue_order:
                glue_object = glue_dict[glue_name]
                glue_obs = glue_object.get_observation(
                    platform_obs, self.observation_space()[platform_name], self.observation_units()[platform_name]
                )
                if glue_obs:
                    platform_obs[glue_name] = glue_obs
                    if glue_object.config.clip_to_space:
                        platform_obs[glue_name] = EnvSpaceUtil.clip_space_sample_to_space(glue_obs, glue_object.observation_space())
            return_observation[platform_name] = [platform_obs]
        return return_observation

    def create_training_observations(self, observations: collections.OrderedDict):
        """
        Base class representing a trainable agent in an environment.
        """

        training_obs: typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]] = collections.OrderedDict()
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

        raw_action_dict: typing.Dict[str, typing.Dict[str, typing.Any]] = collections.OrderedDict()
        obs = self.get_observations()
        for platform_name, platform_action in action.items():
            raw_action_dict[platform_name] = {}
            for glue_name, normalized_action in platform_action.items():
                glue_object = self.agent_glue_dict[platform_name][glue_name]
                raw_action = glue_object.unnormalize_action(normalized_action)
                raw_action_dict[platform_name][glue_name] = raw_action
                if not self.inoperable_platforms[platform_name]:
                    glue_object.apply_action(
                        raw_action,
                        obs,
                        self.action_space()[platform_name],
                        self.observation_space()[platform_name],
                        self.observation_units()[platform_name]
                    )
        return raw_action_dict

    def get_info_dict(self):
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
            for glue_name in self.agent_glue_order:
                glue_info = platform_glues[glue_name].get_info_dict()
                if glue_info:
                    return_info[platform_name][glue_name] = glue_info
        return return_info

    def set_removed(self, removed_state: typing.Dict[str, bool]):
        """
        Set the agent_removed flag for the agent's glues.

        Parameters
        ----------
        - removed_state (typing.Dict[str, bool]): The the inoperablility status of the platform an agent class is controlling.
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

    @lru_cache(maxsize=1)
    def observation_space(self):
        """
        Returns agents obervation space
        """
        tmp = self._create_space(space_getter=lambda glue_obj: glue_obj.observation_space())
        tmp = gym.spaces.Dict({platform_name: Repeated(platform_space, max_len=1) for platform_name, platform_space in tmp.spaces.items()})
        return tmp

    @lru_cache(maxsize=1)
    def normalized_observation_space(self):
        """
        Returns agents normalized obervation space
        """
        tmp = self._create_space(
            space_getter=lambda glue_obj: glue_obj.normalized_observation_space()
            if glue_obj.config.training_export_behavior == TrainingExportBehavior.INCLUDE else None
        )
        tmp = gym.spaces.Dict({platform_name: Repeated(platform_space, max_len=1) for platform_name, platform_space in tmp.spaces.items()})
        if self.config.policy_observation_transformation:
            self._obs_space_transformation = self.config.policy_observation_transformation.transformation(
                tmp, self.config.policy_observation_transformation.config
            )
            return self._obs_space_transformation.output_space

        return tmp
