"""
---------------------------------------------------------------------------

This is a US Government Work not subject to copyright protection in the US.
The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import os
import pickle
from abc import abstractmethod
from datetime import datetime
from pathlib import Path

import flatten_dict
import gymnasium
import numpy as np
from ray.rllib import BaseEnv, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID

from corl.environment.multi_agent_env import ACT3MultiAgentEnv
from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.runners.section_factories.engine.rllib.default_evaluation_callbacks import DefaultEvaluationCallbacks

EPOCH_ZFILL = 6
EVAL_CHECKPOINTS_SUBDIR = "eval_checkpoints"


def set_shareable_info(worker: RolloutWorker, training_iteration: int):
    """Sets worker attributes

    Parameters
    ----------
    worker : RolloutWorker
        The worker, either evaluation worker or (training) worker
    training_iteration : int
        The training iteration
    """
    if worker.config.in_evaluation:
        worker.config["training_iteration"] = training_iteration


class EpisodeArtifactLoggingCallback(DefaultEvaluationCallbacks):
    """
    This is basically the same callback used in the evaluation to same trajectory rollouts.
    The difference is that writing to processed batch seems to mess up training,
    and we changed the naming of the files and where they are written.
    """

    def __init__(self):
        super().__init__()
        self._epoch = 0
        self._episode_counter = 0

    @abstractmethod
    def platform_serializer(self):
        """
        Method specifying the serialize function
        """
        raise NotImplementedError

    def extract_environment_state(self, env_state):  # noqa: PLR6301
        return {}

    def on_evaluate_start(
        self,  # noqa: PLR6301
        *,
        algorithm,
        **kwargs,
    ) -> None:
        """Callback before evaluation starts.

        This method gets called at the beginning of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            kwargs: Forward compatibility placeholder.
        """
        training_iteration = algorithm.training_iteration
        algorithm.evaluation_workers.foreach_worker(lambda worker: set_shareable_info(worker, training_iteration), healthy_only=True)

        # Saves the checkpoint into `eval_checkpoints`
        epoch_str = str(training_iteration).zfill(EPOCH_ZFILL)

        output_path = Path(algorithm._logdir).joinpath(EVAL_CHECKPOINTS_SUBDIR, f"epoch_{epoch_str}")  # noqa: SLF001
        os.makedirs(output_path, exist_ok=True)
        algorithm.save_checkpoint(output_path)

        super().on_evaluate_start(algorithm=algorithm, **kwargs)

    def on_episode_start(
        self,  # noqa: PLR6301
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: dict[PolicyID, Policy],
        episode: EpisodeV2,
        **kwargs,
    ) -> None:
        # At this point, episode.user_data has been initialized. Here we add the initial state.
        env = base_env.get_sub_environments()[episode.env_id]
        episode.user_data["initial_state"] = env.observation
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)

    def on_episode_step(
        self,  # noqa: PLR6301
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: dict[PolicyID, Policy] | None = None,
        episode: EpisodeV2,
        **kwargs,
    ) -> None:
        # Logs the magnitude of each action
        env = base_env.get_sub_environments()[episode.env_id]
        if env.actions:
            # Cycles through each agent
            for agent_id, action_dict in env.actions[-1].items():
                # Flattens the dictionary
                flattened_action_dict = flatten_dict.flatten(action_dict, reducer="dot")
                # Cycles each each action
                for flattened_name, value in flattened_action_dict.items():
                    metric_name = f"{agent_id}/{flattened_name}"
                    episode.custom_metrics[metric_name] = value
                    episode.custom_metrics[f"{metric_name}_mag"] = np.abs(value)

        super().on_episode_step(worker=worker, policies=policies, base_env=base_env, episode=episode, **kwargs)

    def on_episode_end(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: dict[PolicyID, Policy], episode: EpisodeV2, **kwargs
    ) -> None:
        """
        on_episode_end stores the custom metrics in RLLIB. Note this is on a per glue basis.

        1. read the training information for the current episode
        2. For each metric in each platform interface in each environment
           update metric container

        Parameters
        ----------
        worker : RolloutWorker
            Reference to the current rollout worker.
        base_env : BaseEnv
            BaseEnv running the episode. The underlying
            env object can be gotten by calling base_env.get_sub_environments().
        policies : dict
            Mapping of policy id to policy objects. In single
            agent mode there will only be a single "default" policy.
        episode : MultiAgentEpisode
            EpisodeV2 object which contains episode
            state. You can use the `episode.user_data` dict to store
            temporary data, and `episode.custom_metrics` to store custom
            metrics for the episode.
        """

        # Don't need to run if we are not evaluating
        if not worker.config.in_evaluation:
            return

        epoch = str(worker.config["training_iteration"]).zfill(EPOCH_ZFILL)
        env = base_env.get_sub_environments()[episode.env_id]
        done_string_list = []
        # Generate the filenames for trajectories
        for platform_name, done_dict in env.done_info.items():
            for done_name, done_state in done_dict.items():
                if done_state:
                    done_string_list.extend([platform_name[0].upper(), done_name, str(done_state)])
        episode_counter_str = str(self._episode_counter).zfill(8)
        worker_index_str = str(worker.env_context.worker_index).zfill(3)
        done_string = "_".join(done_string_list)
        done_string = f"{episode_counter_str}-{worker_index_str}-{done_string}"

        sub_directory = Path(worker._original_kwargs["log_dir"]) / "trajectories" / f"epoch_{epoch}"  # noqa: SLF001
        policy_artifact = {}
        for policy_id in policies:
            policy_artifact.update(
                {
                    policy_id: EpisodeArtifact.PolicyArtifact(
                        preprocessor=get_preprocessor(env.observation_space)(env.observation_space), filters=worker.filters[policy_id]
                    )
                }
            )
        assert isinstance(worker.env, ACT3MultiAgentEnv)
        assert isinstance(worker.env.action_space, gymnasium.spaces.Dict)
        assert isinstance(worker.env.observation_space, gymnasium.spaces.Dict)
        space_definitions = EpisodeArtifact.SpaceDefinitions(
            action_space=worker.env._raw_action_space,  # noqa: SLF001
            normalized_action_space=worker.env.action_space,
            observation_space=worker.env._raw_observation_space,  # noqa: SLF001
            normalized_observation_space=worker.env.observation_space,
        )
        episode_artifact = EpisodeArtifact(
            test_case=episode.user_data["test_case"],
            env_config=episode.user_data["env_config"],
            artifacts_filenames=episode.user_data["episode_artifacts_filenames"],
            start_time=episode.user_data["start_time"],
            duration_sec=(datetime.now() - episode.user_data["start_time"]).total_seconds(),
            steps=episode.user_data["steps"],
            walltime_sec=episode.user_data["walltime_sec"],
            dones=episode.user_data["dones"],
            frame_rate=env.simulator.frame_rate,
            episode_state=episode.user_data["episode_state"],
            simulator_info=episode.user_data["simulator_info"],
            worker_index=worker.env_context.worker_index,
            params=episode.user_data["params"],
            parameter_values=episode.user_data["parameter_values"],
            observation_units=env._observation_units,  # noqa: SLF001
            platform_to_agents=env.platform_to_agents,
            agent_to_platforms=env.agent_to_platforms,
            done_config=episode.user_data["done_config"],
            initial_state=episode.user_data["initial_state"],
            policy_artifact=policy_artifact,
            space_definitions=space_definitions,
            platform_serializer=self.platform_serializer(),
        )

        if not os.path.exists(sub_directory):
            os.makedirs(sub_directory, exist_ok=True)

        with open(sub_directory / f"{done_string}.pickle", "wb") as handle:
            pickle.dump(episode_artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._episode_counter += 1

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: EpisodeV2,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: dict[AgentID, tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        ...

        # This is an overwrite.
        #
        # Default Evaluation callbacks writes EpisodeArtifact to processed_batch.
        # This creates problems during training when batches need to be concatenated and
        # EpisodeArtifact cannot be concatenated. Only seems to effect single policy
        # where two different platforms contribute to a single Sample Batch
