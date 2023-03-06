"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Callbacks to do logging of episodes
"""
import collections
import copy
import os
import typing
from abc import ABC, abstractmethod
from datetime import datetime

import flatten_dict
import numpy as np
from ray.rllib import BaseEnv, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.simulators.base_simulator import BaseSimulator


class DefaultEvaluationCallbacks(DefaultCallbacks, ABC):
    """Callbacks to extract data for evaluation reporting.

    This is intended to be used as a subclass of the environment callbacks used by the experiment using cooperative multiple inheritance.
    """

    @abstractmethod
    def platform_serializer(self):
        """
        Function that holds a placeholder for the SerializePlatform object which contains a serialization function.
        The SerializePlatform object is to be instantiated and placed inside this method for future use
        """

    @abstractmethod
    def extract_environment_state(self, env_state: dict):
        """Method that extracts any information in the environment state that needs to be saved
        """
        return {}

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: typing.Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        """See corl.environment.default_env_rllib_callbacks.EnvironmentDefaultCallbacks.on_episode_start"""

        env = base_env.get_sub_environments()[episode.env_id]

        episode.user_data['eval'] = {
            agent_id: {
                'episode': {
                    'done': {}, 'reward': {}, 'total_reward': np.nan, 'episode_state': {}
                }, 'step': []
            }
            for agent_id in env.agent_dict.keys()
        }

        episode.user_data['test_case'] = env.episode_id
        episode.user_data['params'] = flatten_dict.flatten(env.local_variable_store, reducer="dot")
        episode.user_data['start_time'] = datetime.now()
        episode.user_data['episode_artifacts_filenames'] = {}
        episode.user_data['observation_units'] = env._observation_units  # pylint: disable=protected-access
        episode.user_data['platform_to_agents'] = env.platform_to_agents
        episode.user_data['agent_to_platforms'] = env.agent_to_platforms
        episode.user_data['done_config'] = EpisodeArtifact.DoneConfig(task=env.config.dones.task, world=env.config.dones.world)

        step_data: typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]] = {
            agent_id: {
                'observation': {}, 'action': {}
            }
            for agent_id in env.agent_dict.keys()
        }

        for agent_id, obj_pair in env.observation.items():
            step_data[agent_id]['observation'] = obj_pair

        for agent_id in env.agent_dict.keys():  # pylint: disable=protected-access
            episode.user_data['eval'][agent_id]['step'].append(step_data[agent_id])

        episode.user_data["steps"] = []
        episode.user_data["dones"] = {}

        # collect and add full set of parameter values for episode
        parameter_values = {'environment': flatten_dict.flatten(env.local_variable_store, reducer="dot")}
        for agent_id, agent in env.agent_dict.items():
            parameter_values[agent_id] = flatten_dict.flatten(agent.local_variable_store, reducer="dot")
        episode.user_data["parameter_values"] = parameter_values

        # Cooperative multiple inheritance
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)

    # pylint: disable=too-many-branches
    # # pylint: disable=too-many-statements
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: typing.Optional[typing.Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:
        """See corl.environment.default_env_rllib_callbacks.EnvironmentDefaultCallbacks.on_episode_step"""

        env = base_env.get_sub_environments()[episode.env_id]

        episode.user_data['eval'] = {
            agent_id: {
                'episode': {
                    'done': {}, 'reward': {}, 'total_reward': np.nan, 'episode_state': {}
                }, 'step': []
            }
            for agent_id in env.agent_dict.keys()
        }

        # # Calculate this step's rewards
        map_agent_reward: typing.Dict[str, typing.Dict[str, float]] = {}
        if env.reward_info is not None:
            for agent_id, obj in env.reward_info.items():
                for name, value in obj.items():
                    if agent_id not in map_agent_reward:
                        map_agent_reward[agent_id] = {}
                    if '__all__' in agent_id:
                        continue
                    map_agent_reward[agent_id].update(self._resolve_sequence_values(name, value))

        # Calculate this step's done
        if env.done_info is not None:
            for agent_id, obj in env.done_info.items():
                for name, value in obj.items():
                    if agent_id not in episode.user_data["dones"]:
                        episode.user_data["dones"][agent_id] = {}
                    if '__all__' in agent_id:
                        continue
                    episode.user_data["dones"][agent_id].update(self._resolve_sequence_values(name, value))

        episode_state: dict = copy.deepcopy(env.state.agent_episode_state)

        # ######################################
        # # Step data - append on each step
        step_data: typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]] = {
            agent_id: {
                'observation': {}, 'action': {}
            }
            for agent_id in env.agent_dict.keys()
        }

        for agent_id, obj_pair in env.observation.items():
            step_data[agent_id]['observation'] = obj_pair

        if env._actions:  # pylint: disable=protected-access, too-many-nested-blocks
            for agent_id, agent_data in env._actions[-1].items():  # pylint: disable=protected-access

                agent_data = flatten_dict.flatten(agent_data)
                for part_name, value in agent_data.items():
                    part_name = '.'.join(part_name)
                    if isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], dict):
                        # handle RTAModule or otherwise wrapped controllers
                        for elem in value:
                            for sub_part_name, sub_value in elem.items():
                                full_part_name = part_name + "." + sub_part_name
                                step_data[agent_id]['action'][full_part_name] = sub_value
                    else:
                        step_data[agent_id]['action'][part_name] = value

                # step_data[agent_id]['action'] = agent_data    # change from corl2.0 merge

        # ##################################################
        # # Append the episode's step information to our agent_data knowledge
        serialized_platforms = [self.platform_serializer().serialize(item) for item in env.state.sim_platforms.values()]

        # # Start our step artifact, the agent map will be filled out next
        step = EpisodeArtifact.Step(agents={}, platforms=serialized_platforms, environment_state=self.extract_environment_state(env.state))

        for agent_id, trainable_agent in env.agent_dict.items():  # pylint: disable=protected-access
            policy_id = episode.policy_for(agent_id)

            # If the platform doesn't exists anymore there is nothing to do.
            # # Assumes 1:1 map of agent to platform
            if len([item for item in serialized_platforms if item["name"] == trainable_agent.platform_names[0]]) != 0:
                step.agents[agent_id] = EpisodeArtifact.AgentStep(
                    step_data[agent_id]["observation"],
                    step_data[agent_id]["action"],
                    map_agent_reward.get(agent_id),
                    episode.agent_rewards[(agent_id, policy_id)]
                )

        episode.user_data["steps"].append(step)
        episode.user_data["episode_state"] = episode_state

        #####################################################
        # Legacy eval setting - overwrite on each step so only last one persists

        for agent_id in env.agent_dict.keys():
            episode.user_data['eval'][agent_id]['step'].append(step_data[agent_id])

        episode.user_data['eval'][agent_id]['episode']['done'] = episode.user_data["dones"]

        episode.user_data['eval'][agent_id]['episode']['reward'] = map_agent_reward

        for agent_id in env.agent_dict.keys():
            policy_id = episode.policy_for(agent_id)
            episode.user_data['eval'][agent_id]['episode']['total_reward'] = episode.agent_rewards[(agent_id, policy_id)]

        episode.user_data['eval'][agent_id]['episode']['episode_state'] = episode_state[agent_id]

        # Cooperative multiple inheritance
        super().on_episode_step(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: typing.Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: typing.Dict[AgentID, typing.Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        """See corl.environment.default_env_rllib_callbacks.EnvironmentDefaultCallbacks.on_postprocess_trajectory"""

        if 'eval_legacy' not in postprocessed_batch:
            postprocessed_batch['eval_legacy'] = [{'test_case': episode.user_data['test_case'], 'runtime': np.array([-1]), 'eval_data': {}}]

        if worker.env is None:
            raise RuntimeError("Worker.env is None, this is a non op")
        if not isinstance(worker.env.simulator, BaseSimulator):
            raise RuntimeError("The simulator attached to the worker does not inherit BaseSimulator")
        frame_rate = worker.env.simulator.frame_rate

        # If we have yet to generate the episode artifact for this episode then do it
        if 'episode_artifact' not in episode.user_data:
            episode.user_data['episode_artifact'] = np.array(
                [
                    EpisodeArtifact(
                        test_case=episode.user_data['test_case'],
                        artifacts_filenames=episode.user_data['episode_artifacts_filenames'],
                        wall_time_sec=(datetime.now() - episode.user_data['start_time']).total_seconds(),
                        steps=episode.user_data["steps"],
                        dones=episode.user_data["dones"],
                        frame_rate=frame_rate,
                        episode_state=episode.user_data["episode_state"],
                        worker_index=worker.env_context.worker_index,
                        params=episode.user_data["params"],
                        parameter_values=episode.user_data["parameter_values"],
                        observation_units=episode.user_data["observation_units"],
                        platform_to_agents=episode.user_data["platform_to_agents"],
                        agent_to_platforms=episode.user_data["agent_to_platforms"],
                        done_config=episode.user_data["done_config"]
                    )
                ]
            )

        if 'eval' not in postprocessed_batch:
            postprocessed_batch['eval'] = episode.user_data['episode_artifact']

        postprocessed_batch['eval_legacy'][0]['agent_id'] = agent_id
        postprocessed_batch['eval_legacy'][0]['runtime'] = (datetime.now() - episode.user_data['start_time']).total_seconds()
        postprocessed_batch['eval_legacy'][0]['eval_data'][agent_id] = episode.user_data['eval'][agent_id]
        # postprocessed_batch['eval'][0]['episode_state'][agent_id] = episode.user_data['eval'][agent_id]
        postprocessed_batch['eval_legacy'][0]['pid'] = os.getpid()

        # Cooperative multiple inheritance
        super().on_postprocess_trajectory(
            worker=worker,
            episode=episode,
            agent_id=agent_id,
            policy_id=policy_id,
            policies=policies,
            postprocessed_batch=postprocessed_batch,
            original_batches=original_batches,
            **kwargs
        )

    def _resolve_sequence_values(self, key: str, value: typing.Any) -> typing.Dict[str, typing.Any]:
        """Convert a single key with a sequence of values into a sequence of keys each with a single value.

        Parameters
        ----------
        key : str
            Base key for this value
        value : Any
            Value for this key.  This can be a single value, such as a string or number, or a sequence of values.

        Returns
        -------
        Dict[str, Any]
            A dictionary with keys that are the provided key with index values concatenated.  The values are the sequence values at that
            index.
        Example
        >>> self.__resolve_sequence_values('foo', 'bar')
        {'foo': 'bar'}
        >>> self.__resolve_sequence_values('foo', [1, 2, 3])
        {'foo_0': 1, 'foo_1': 2, 'foo_2': 3}
        """
        output: typing.Dict[str, typing.Any] = {}
        if isinstance(value, str):
            output[key] = value
        elif isinstance(value, (collections.abc.Sequence, np.ndarray)):
            if isinstance(value, np.ndarray) and not value.shape:
                output[key] = value.item()
            elif len(value) > 1:
                for index, new_value in enumerate(value):
                    output.update(self._resolve_sequence_values(f'{key}_{index}', new_value))
            elif len(value) == 1:
                output.update(self._resolve_sequence_values(key, value[0]))
            else:
                output[key] = 'empty_list'
        elif isinstance(value, collections.abc.Mapping):
            for key2, value2 in value.items():
                output.update(self._resolve_sequence_values(f'{key}.{key2}', value2))
        else:
            output[key] = value
        return output
