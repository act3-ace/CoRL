"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
EnvironmentDefaultCallbacks
"""
# pylint: disable=arguments-differ, W0613
import typing
import warnings
from collections import defaultdict

import numpy as np
from flatten_dict import flatten
from gym.utils import seeding
from ray.rllib import BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID

from corl.dones.done_func_base import DoneStatusCodes
from corl.environment.multi_agent_env import ACT3MultiAgentEnv

SHORT_EPISODE_THRESHOLD = 5


def log_done_status(env, episode):
    """
    Log done status codes to done_status/{platform}/{status}
    """

    def process_done_entry(platform, data, prefix=""):
        if not data:
            return
        at_least_table = [0] * len(DoneStatusCodes)
        episode_codes = set(data.values())

        for indx, status in enumerate(DoneStatusCodes):
            metric_key = f"{prefix}done_status/{platform}/{status}"
            if status in episode_codes:
                episode.custom_metrics[metric_key] = 1
                for x in range(indx, len(at_least_table)):
                    at_least_table[x] += 1
            else:
                episode.custom_metrics[metric_key] = 0

        for indx, status in enumerate(DoneStatusCodes):
            if indx in [0, 4]:
                continue
            metric_key = f"{prefix}done_status/{platform}/AT_LEAST_{status}"
            episode.custom_metrics[metric_key] = at_least_table[indx]

    if env.state.episode_state:
        for platform, data in env.state.episode_state.items():
            process_done_entry(platform=platform, data=data)

    if env.state.agent_episode_state:
        for agent, data in env.state.agent_episode_state.items():
            process_done_entry(platform=agent, data=data, prefix="agent_")


def log_done_info(env, episode):
    """
    Log done info to done_results/{platform}/{done_name}
    """
    if env.done_info:
        for platform_id, platform_data in env.done_info.items():
            for done_name, done_value in platform_data.items():
                episode.custom_metrics[f'done_results/{platform_id}/{done_name}'] = int(done_value)
            episode.custom_metrics[f'done_results/{platform_id}/NoDone'] = int(not any(platform_data.values()))


class EnvironmentDefaultCallbacks(DefaultCallbacks):
    """
    This is the default class for callbacks to be use in the Environment class.
    To make your own custom callbacks set the EnvironmentCallbacks in your derived class
    to be a class that subclasses this class (EnvironmentDefaultCallbacks)

    Make sure you call the super function for all derived functions or else there will be unexpected callback behavior
    """

    DEFAULT_METRIC_OPS = {
        "min": np.min,
        "max": np.max,
        "median": np.median,
        "mean": np.mean,
        "var": np.var,
        "std": np.std,
        "sum": np.sum,
        "nonzero": np.count_nonzero,
    }

    def on_episode_start(
        self,
        *,
        worker,
        base_env: BaseEnv,
        policies: typing.Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        """Callback run on the rollout worker before each episode starts.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode: Episode object which contains the episode's
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)

        episode.user_data["rewards_accumulator"] = defaultdict(float)  # default dict with default value of 0.0

    def on_episode_step(
        self,
        *,
        worker,
        base_env: BaseEnv,
        policies: typing.Optional[typing.Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """

        super().on_episode_step(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)

        env = base_env.get_sub_environments()[episode.env_id]

        if env.reward_info:
            rewards_accumulator = episode.user_data["rewards_accumulator"]
            for reward_name, reward_val in flatten(env.reward_info, reducer="path").items():
                key = f"rewards_cumulative/{reward_name}"
                rewards_accumulator[key] += reward_val

    def on_episode_end(  # pylint: disable=too-many-branches
        self,
        *,
        worker,
        base_env: BaseEnv,  # pylint: disable=too-many-branches
        policies: typing.Dict[PolicyID, Policy],
        episode,
        **kwargs
    ) -> None:
        """
        on_episode_end  stores the custom metrics in RLLIB. Note this is on a per glue basis.

        1. read the training information for the current episode
        2. For each metric in each platform interface in each environment
           update metric container

        Parameters
        ----------
        worker: RolloutWorker
            Reference to the current rollout worker.
        base_env: BaseEnv
            BaseEnv running the episode. The underlying
            env object can be gotten by calling base_env.get_sub_environments().
        policies: dict
            Mapping of policy id to policy objects. In single
            agent mode there will only be a single "default" policy.
        episode: MultiAgentEpisode
            Episode object which contains episode
            state. You can use the `episode.user_data` dict to store
            temporary data, and `episode.custom_metrics` to store custom
            metrics for the episode.
        """

        # Issue a warning if the episode is short
        # This, should get outputing in the log and may help identify any setup issues
        if episode.length < SHORT_EPISODE_THRESHOLD:
            msg = f"Episode {str(episode.episode_id)} length {episode.length} is less than warn threshold {str(SHORT_EPISODE_THRESHOLD)}"
            if "params" in episode.user_data:
                msg += "\nparams:\n"
                for key in episode.user_data["params"].keys():
                    msg += f"  {key}: {str(episode.user_data['params'][key])}"
                    msg += "\n"
            else:
                msg += "\nParams not provied in episode user_data"
            warnings.warn(msg)

        env = base_env.get_sub_environments()[episode.env_id]
        if env.glue_info:  # pylint: disable=too-many-nested-blocks
            for glue_name, metric_val in flatten(env.glue_info, reducer="path").items():
                episode.custom_metrics[glue_name] = metric_val

        if env.reward_info:
            for reward_name, reward_val in flatten(env.reward_info, reducer="path").items():
                key = f"rewards/{reward_name}"
                episode.custom_metrics[key] = reward_val

        log_done_info(env, episode)

        log_done_status(env, episode)

        # Variables
        for key, value in flatten(env.local_variable_store, reducer="path").items():
            try:
                episode.custom_metrics[f'variable/env/{key}'] = float(value.value)
            except ValueError:
                pass

        for agent_name, agent_data in env.agent_dict.items():
            for key, value in flatten(agent_data.local_variable_store, reducer="path").items():
                try:
                    episode.custom_metrics[f'variable/{agent_name}/{key}'] = float(value.value)
                except ValueError:
                    pass

        # Episode Parameter Providers
        metrics = env.config.epp.compute_metrics()
        for k, v in metrics.items():
            episode.custom_metrics[f'adr/env/{k}'] = v

        for agent_name, agent_data in env.agent_dict.items():
            metrics = agent_data.config.epp.compute_metrics()
            for k, v in metrics.items():
                episode.custom_metrics[f'adr/{agent_name}/{k}'] = v

        # Cumulative Rewards
        for key, value in episode.user_data["rewards_accumulator"].items():
            episode.custom_metrics[key] = value

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: typing.Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: typing.Dict[AgentID, typing.Tuple[Policy, SampleBatch]],
        **kwargs
    ) -> None:
        """
        Called immediately after a policy's postprocess_fn is called.

        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.

        Parameters
        ----------
        worker: RolloutWorker
            Reference to the current rollout worker.
        episode: MultiAgentEpisode
            Episode object.
        agent_id: str
            Id of the current agent.
        policy_id: str
            Id of the current policy for the agent.
        policies: dict
            Mapping of policy id to policy objects. In single
            agent mode there will only be a single "default" policy.
        postprocessed_batch: SampleBatch
            The postprocessed sample batch
            for this agent. You can mutate this object to apply your own
            trajectory postprocessing.
        original_batches: dict
            Mapping of agents to their unpostprocessed
            trajectory data. You should not mutate this object.
        """
        super().on_postprocess_trajectory(
            worker=worker,
            episode=episode,
            agent_id=agent_id,
            policy_id=policy_id,
            policies=policies,
            postprocessed_batch=postprocessed_batch,
            original_batches=original_batches
        )
        episode.worker.foreach_env(
            lambda env: env.post_process_trajectory(agent_id, postprocessed_batch, episode, policies[policy_id])
        )  # type: ignore

    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        """
        Called at the end of Trainable.train().

        Parameters
        ----------
        algorithm: algorithm
            Current algorithm instance.
        result: dict
            Dict of results returned from algorithm.train() call.
            You can mutate this object to add additional metrics.
        """
        rng, _ = seeding.np_random(seed=algorithm.iteration)

        # Environment EPP
        assert result['config']['env'] == ACT3MultiAgentEnv

        for epp in result['config']['env_config']['epp_registry'].values():
            epp.update(result, rng)
