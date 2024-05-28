# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
"""
Extra callbacks that can be for evaluating during training to
generate episode artifacts.
"""

import copy
import typing
from collections import OrderedDict

from ray.rllib.utils.typing import MultiAgentDict

import corl.experiments.rllib_utils.wrappers as rllib_wrappers
from corl.agents.base_agent import BaseAgent
from corl.environment.base_multi_agent_env import BaseCorlMultiAgentEnv, BaseCorlMultiAgentEnvValidator
from corl.environment.multi_agent_env import ACT3MultiAgentEnv
from corl.episode_parameter_providers import EpisodeParameterProvider


class CorlEnvWrapperValidator(BaseCorlMultiAgentEnvValidator):
    """Validation model for the inputs of CorlGroupedAgentEnv"""

    wrapped_env: str = "CorlMultiAgentEnv"


class CorlEnvWrapper(BaseCorlMultiAgentEnv):
    """Environment that wraps another CoRL environment."""

    def __init__(self, config: dict):
        """Initialize CorlEnvWrapper."""
        self.config: CorlEnvWrapperValidator
        super().__init__(config)
        self._wrapped_env: ACT3MultiAgentEnv = rllib_wrappers.get_rllib_environment_creator(self.config.wrapped_env)(config)
        self._agent_dict = self._wrapped_env.agent_dict
        self._simulator = self._wrapped_env.simulator
        self._state = self._wrapped_env.state
        self.observation_space = self._wrapped_env.observation_space
        self.action_space = self._wrapped_env.action_space

    @staticmethod
    def get_validator() -> type[CorlEnvWrapperValidator]:
        """Get the validator for this class."""
        return CorlEnvWrapperValidator

    def post_process_trajectory(self, agent_id, batch, episode, policy):
        """easy accessor for calling post process trajectory
            correctly

        Arguments:
            agent_id: agent id
            batch: post processed Batch - be careful modifying
        """
        raise NotImplementedError

    @property
    def local_variable_store(self) -> dict[str, typing.Any]:
        """
        get the local variable store

        Returns
        -------
        Dict
            the local variable store episode id
        """
        return self._wrapped_env.local_variable_store

    @property
    def epp(self) -> EpisodeParameterProvider:
        """
        get the Episode Parameter Provider

        Returns
        -------
        EpisodeParameterProvider
        """
        return self._wrapped_env.epp

    @property
    def epp_registry(self) -> dict[str, EpisodeParameterProvider]:
        """
        get the Episode Parameter Provider Registry

        Returns
        -------
        dict[str, EpisodeParameterProvider]
        """
        return self._wrapped_env.epp_registry

    @property
    def corl_agent_dict(self) -> dict[str, BaseAgent]:
        """
        get agents in ACT3MultiAgentEnv

        Returns
        -------
        dict[str, BaseAgent]
        """
        return self._wrapped_env.corl_agent_dict

    def reset(self, *, seed=None, options=None) -> tuple[MultiAgentDict, MultiAgentDict]:
        """Reset env"""
        return_data = self._wrapped_env.reset(seed=seed, options=options)
        self._update_properties()
        return return_data

    def step(self, action_dict: dict):
        """Reset env"""
        step_return = self._wrapped_env.step(action_dict=action_dict)
        self._update_properties()
        return step_return

    def _update_properties(self):
        self._actions = self._wrapped_env.actions
        self._observation = self._wrapped_env.observation
        self._info = self._wrapped_env.info
        self._done_info = self._wrapped_env.done_info
        self._reward_info = self._wrapped_env.reward_info
        self._episode_id = self._wrapped_env.episode_id
        self._state = self._wrapped_env.state


class GroupedAgentsEnvValidator(CorlEnvWrapperValidator):
    """Validation model for the inputs of GroupedAgentsEnv"""

    groups: dict[str, list]


class GroupedAgentsEnv(CorlEnvWrapper):
    """Wrapping environment that combines groups of agents into single agents"""

    def __init__(self, config: dict):
        """Initialize GroupedAgentsEnv."""
        self.config: GroupedAgentsEnvValidator
        super().__init__(config)

        self._groups = self.config.groups
        self._agent_id_to_group = self._create_agent_id_map(self._groups)
        self._info = self._group_items(self._wrapped_env.info)
        self._agent_dict = self._group_items(self._wrapped_env.agent_dict)
        self.action_space = self._group_items(self._wrapped_env.action_space)
        self.observation_space = self._group_items(self._wrapped_env.observation_space)
        self._observation_units = self._group_items(self._wrapped_env.observation_units)

        self._agent_to_platforms = {group: [] for group in self._groups}
        for agent, platforms in self._wrapped_env.agent_to_platforms.items():
            self._agent_to_platforms[self._agent_id_to_group[agent]].append(platforms)

        self._platform_to_agents = {platform: [] for platform in self._wrapped_env.platform_to_agents}
        for platform, agents in self._wrapped_env.platform_to_agents.items():
            for agent in agents:
                self._platform_to_agents[platform].append(self._agent_id_to_group[agent])

    @staticmethod
    def get_validator() -> type[GroupedAgentsEnvValidator]:
        """Get the validator for this class."""
        return GroupedAgentsEnvValidator

    def reset(self, *, seed=None, options=None) -> tuple[MultiAgentDict, MultiAgentDict]:
        """Reset env"""
        obs, trainable_info = self._wrapped_env.reset(seed=seed, options=options)
        self._observation = self._group_items(obs)
        self.actions.clear()

        return self.observation, self._group_items(trainable_info)

    def step(self, action_dict: dict):
        """Step env"""
        obs, rewards, dones, truncated_dones, info = self._wrapped_env.step(action_dict=self._ungroup_items(action_dict))

        training_obs = self._group_items(obs)
        grouped_rewards = self._group_items(rewards, agg_fn=self._ave)
        grouped_dones = self._group_items(dones, agg_fn=self._all)
        grouped_truncated_dones = self._group_items(truncated_dones, agg_fn=self._all)
        grouped_trainable_info = self._group_items(info)

        if not self._wrapped_env.config.disable_action_history:
            self.actions.append(action_dict)
        self._update_properties()

        return training_obs, grouped_rewards, grouped_dones, grouped_truncated_dones, grouped_trainable_info

    def post_process_trajectory(self, agent_id, batch, episode, policy):
        """easy accessor for calling post process trajectory
            correctly

        Arguments:
            agent_id: agent id
            batch: post processed Batch - be careful modifying
        """

    def _create_agent_dict(self) -> dict[str, BaseAgent]:
        """
        get the agent dict

        Returns
        -------
        Dict
            agents
        """
        grouped_agent_dict = self._group_items(self._wrapped_env.agent_dict)
        for group in list(grouped_agent_dict.keys()):
            for agent_name, agent in grouped_agent_dict[group].items():
                grouped_agent_dict[f"{group}/{agent_name}"] = agent
            grouped_agent_dict.pop(group)

        return grouped_agent_dict

    def _update_properties(self):
        self._observation = self._group_items(self._wrapped_env.observation)
        self._info = self._group_items(self._wrapped_env.info)
        self._done_info = self._group_items(self._wrapped_env.done_info)
        self._reward_info = self._group_items(self._wrapped_env.reward_info)
        self._episode_id = self._wrapped_env.episode_id
        self._state = copy.deepcopy(self._wrapped_env.state)
        self._state.episode_state = self._group_items(self._wrapped_env.state.episode_state)
        for agent_name in self._agent_dict:
            self._state.agent_episode_state[agent_name] = OrderedDict()

        for platform_name in self.state.episode_state:
            for agent_name in self.platform_to_agents.get(platform_name, []):
                for done_name, done_status in self._state.episode_state[platform_name].items():
                    self._state.agent_episode_state[agent_name][done_name] = done_status

    def _ungroup_items(self, grouped_dict):
        ungrouped_dict = type(grouped_dict)()
        for group_name, agents in self._groups.items():
            for agent in agents:
                ungrouped_dict[agent] = grouped_dict[group_name][agent]
        return ungrouped_dict

    @staticmethod
    def _create_agent_id_map(groups):
        agent_id_to_group = {}
        for group_id, agent_ids in groups.items():
            for agent_id in agent_ids:
                if agent_id in agent_id_to_group:
                    raise ValueError(f"Agent id {agent_id} is in multiple groups")
                agent_id_to_group[agent_id] = group_id
        return agent_id_to_group

    def _group_items(self, ungrouped_items, agg_fn=lambda x: x):
        grouped_items = type(ungrouped_items)()
        for agent_id, item in ungrouped_items.items():
            if agent_id in self._agent_id_to_group:
                group_id = self._agent_id_to_group[agent_id]
                if group_id in grouped_items:
                    continue  # already added

                group_out = type(ungrouped_items)()
                for agent in self._groups[group_id]:
                    group_out[agent] = ungrouped_items[agent]

                grouped_items[group_id] = agg_fn(group_out)
            else:
                grouped_items[agent_id] = item
        return grouped_items

    @staticmethod
    def _ave(grouped_items: dict):
        """returns average of value in dict"""
        values = grouped_items.values()
        return sum(values) / len(values)

    @staticmethod
    def _all(grouped_items: dict):
        """returns true if all values in dict are true otherwise false"""
        return all(grouped_items.values())
