# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

import glob
import logging
import math
import os
import secrets
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import numpy as np
import ray.cloudpickle as pickle
from flatten_dict import flatten_dict
from pydantic import BaseModel
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline import InputReader, IOContext
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch, concat_samples
from ray.rllib.utils.annotations import override
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.typing import SampleBatchType

from corl.environment.base_multi_agent_env import BaseCorlMultiAgentEnv
from corl.evaluation.episode_artifact import EpisodeArtifact

AGENT_INDEX_VALUE = 0

logger = logging.getLogger(__name__)


class EpisodeArtifactReaderInputConfigValidator(BaseModel):
    """
    Validator for ioctx.input_config

    Parameters
    ----------
    inputs: str | list | tuple
        Either a path to a directory (str) where pickle files
        are stored, a single file (str) or a list or tuple of individual files.
    agemt_id_in_artifacts: str
        The name of the agent that we are reading offline data
        from. That is, this is the key correspond to the agent name in the episode artifacts.

    """

    inputs: str | list | tuple
    agent_id_in_artifacts: str = DEFAULT_POLICY_ID


class EpisodeArtifactReader(InputReader):
    """Reader loads episode artifacts and transforms them into
    SampleBatches.

    Like the JsonReader, this reader only supports reading offline data for
    one policy where the policy_id must be `DEFAULT_POLICY_ID="default_policy"`

    To include the reader in your offline experiment, including the following
    within the rllib_config.yml:

    ```yaml
    input_: corl.evaluation.runners.section_factories.engine.rllib.offline_rl.episode_artifact_reader.EpisodeArtifactReader
    input_config:
        inputs: /path/to/episode_artifacts/directory/
        agent_id_in_artifacts: paddle0_ctrl (optional)
    ```

    NOTE: See docstring for `EpisodeArtifactReaderInputConfigValidator` for more details.
    """

    def __init__(self, ioctx: IOContext) -> None:
        super().__init__()

        # #####################################################################
        # This is mostly taken from JsonReader, with some slight modifications
        # #####################################################################

        self.ioctx = ioctx or IOContext()
        input_config = EpisodeArtifactReaderInputConfigValidator(**self.ioctx.input_config)
        self.agent_id = input_config.agent_id_in_artifacts
        self.batch_size = int(self.ioctx.config.get("train_batch_size", 1))
        num_workers = self.ioctx.config.get("num_workers", 0)
        if num_workers:
            self.batch_size = max(math.ceil(self.batch_size / num_workers), 1)
        assert self.ioctx.worker is not None, "Rollout worker is missing from IOContext"

        self.policy_map = self.ioctx.worker.policy_map
        assert self.policy_map is not None, "Policy map is missing from the worker."
        self.default_policy = self.policy_map[DEFAULT_POLICY_ID]

        # Input is a string
        if isinstance(input_config.inputs, str):
            abs_path = os.path.abspath(os.path.expanduser(input_config.inputs))
            # The string is a directory
            if os.path.isdir(abs_path):
                inputs = [os.path.join(abs_path, "*.p*kl*")]
                logger.warning("Treating input directory as glob patterns: %s", inputs)
            # The string is a single file
            else:
                inputs = [abs_path]
            self.files = []
            for i in inputs:
                self.files.extend(glob.glob(i))

        # When the input is a list or tuple
        elif isinstance(input_config.inputs, list | tuple):
            self.files = list(input_config.inputs)

        if self.files:
            logger.info("Found %s input files.", len(self.files))
        else:
            raise ValueError(f"No files found matching {input_config.inputs}")

        self.cur_file: str | None = None

    @override(InputReader)
    def next(self) -> SampleBatchType:  # noqa: A003
        """Returns a batch of sample experiences"""
        batch_counter = 0
        assert self.ioctx.worker is not None
        assert isinstance(self.ioctx.worker.env, BaseCorlMultiAgentEnv)
        agent_class = self.ioctx.worker.env.agent_dict[DEFAULT_POLICY_ID]

        assert self.default_policy is not None
        assert not self.default_policy.config["in_evaluation"], (
            "Evaluation using offline data is not supported.",
            "See RLlib documentation on how to evaluate.",
            "https://docs.ray.io/en/latest/rllib/rllib-offline.html#off-policy-estimation-ope",
        )
        sample_batches = []
        # #####################################################################
        # Collect episode artifacts until we have reached the batch size
        # #####################################################################
        while batch_counter < self.batch_size:
            episode = self._load_next_file()
            # The episode ID will correspond to the first timestamp (walltime)
            # in the episode
            episode_id = int(episode.walltime_sec[0])

            init_obs = self.process_observations(episode.initial_state[self.agent_id])
            assert self.policy_map, "Policy map is undefined"
            assert DEFAULT_POLICY_ID in self.policy_map, f"{DEFAULT_POLICY_ID} is missing from the policy_map"

            assert agent_class.trainable, (
                f"Non-trainable agent: {DEFAULT_POLICY_ID} selected!",
                "Check to make sure the policy_id is set correctly.",
            )
            # The collector is intended to collect samples for
            # one agent in one trajectory. here we use it for creating
            # a collection of sample batches from episode artifacts.
            collector = AgentCollector(
                view_reqs=self.default_policy.view_requirements,
                max_seq_len=self.default_policy.max_seq_len,
                is_policy_recurrent=self.default_policy.is_recurrent(),
                is_training=True,
            )
            collector.add_init_obs(init_obs=init_obs, episode_id=episode_id, agent_index=AGENT_INDEX_VALUE, env_id=str(self.cur_file))
            # #################################################################
            #  Processes a single episode
            # #################################################################
            for idx, step in enumerate(episode.steps):
                agent_step: EpisodeArtifact.AgentStep = step.agents[self.agent_id]
                # If there are no observations, continue
                if not agent_step.observations:
                    continue

                array_obs = self.process_observations(agent_step.observations)

                # sums the rewards when they exist
                rewards = None
                if agent_step.rewards:
                    rewards = sum([reward_value for _, reward_value in agent_step.rewards.items()])
                array_actions = self._build_and_process_action_array(agent_step.actions)
                input_values = {
                    SampleBatch.AGENT_INDEX: AGENT_INDEX_VALUE,
                    SampleBatch.ACTIONS: array_actions,
                    SampleBatch.NEXT_OBS: array_obs,
                    SampleBatch.REWARDS: rewards,
                    # Sets TERMINANDS and TRUNCATEDS to 1 at the last step
                    SampleBatch.TERMINATEDS: 1 if idx == len(episode.steps) - 1 else 0,
                    SampleBatch.TRUNCATEDS: 1 if idx == len(episode.steps) - 1 else 0,
                    SampleBatch.T: idx,
                    # While not used here, INFOS is introduced in the AgentCollector.
                    # When excluded, the dimension of SampleBatch.INFOS is mismatched
                    # with the rest of the data and this causes issues down the line.
                    SampleBatch.INFOS: {},
                }

                collector.add_action_reward_next_obs(input_values)
                batch_counter += 1
            # Builds the sample batch for training
            one_episode_sample_batch = collector.build_for_training(view_requirements=self.default_policy.view_requirements)
            sample_batches.append(self.default_policy.postprocess_trajectory(one_episode_sample_batch))

        return concat_samples(sample_batches)

    def _build_and_process_action_array(self, action_dict: dict[str, int | float]) -> np.ndarray:
        """Builds the action array by restructring the action dict into an
        array. The order of the elements follow the policy action dict struct"""
        assert self.default_policy is not None
        # #####################################################################
        # Postprocess Actions
        # -----------------------------------------------------------------
        # This follows how actions are processed by rllib w/ the JsonReader via
        # the postprocess_actions method.
        # When actions_in_input_normalize = False and
        # normalize_actions = True, postprocess_actions will normalize the
        # actions.
        # #####################################################################
        # NOTE: the action space associated with the default policy is dependent
        # on whether or not the glues are normalized. That is, if the controller
        # glues have normalization enabled, the default_policy.action_space
        # will be the normalized action space.
        flattened_action_space = flatten_dict.flatten(self.default_policy.action_space_struct, reducer="dot")
        if self.default_policy.config.get("normalize_actions") and self.default_policy.config.get("actions_in_input_normalized") is False:
            action_dict = normalize_action(action=action_dict, action_space_struct=flattened_action_space)

        control_values: list[float | int] = []
        for controller_name in flattened_action_space:
            control_value = action_dict[controller_name]
            # Process iterables
            if isinstance(control_value, Iterable):
                control_value = list(control_value)  # type: ignore
                control_values.extend(control_value)
            else:
                control_values.append(control_value)
        return np.array(control_values)

    def _load_next_file(self) -> EpisodeArtifact:
        # If this is the first time, we open a file, make sure all workers
        # start with a different one if possible.
        if self.cur_file is None and self.ioctx.worker is not None:
            idx = self.ioctx.worker.worker_index
            total = self.ioctx.worker.num_workers or 1
            path = self.files[round((len(self.files) - 1) * (idx / total))]
        # After the first file, pick all others randomly.
        else:
            path = secrets.choice(self.files)
        self.cur_file = path
        with open(path, "rb") as file_obj:
            return pickle.load(file_obj)

    def process_observations(self, observations: Any) -> Any:
        """Processes observations by:
        1. Filtering down the observations to only those included during training
        2. Reordering the obs according to how they are ordered in the preprocessor
        3. Transforming the obs with the preprocessor

        """
        assert self.ioctx.worker is not None
        assert isinstance(self.ioctx.worker.env, BaseCorlMultiAgentEnv)
        agent_class = self.ioctx.worker.env.agent_dict[DEFAULT_POLICY_ID]
        observation_space = self.ioctx.worker.env.observation_space[DEFAULT_POLICY_ID]
        preprocessor = get_preprocessor(observation_space)(observation_space)

        # Filters out observations that are not used for training
        prepped_obs = agent_class.create_training_observations(observations)
        # We need to reorder the observations according to the observation space in the preprocessor because
        # RLlib uses this ordering to create its list of preprocessors. When the order is mismatched, the preprocessors are mismatched
        if isinstance(prepped_obs, OrderedDict):
            prepped_obs = reorder_observations(observations_dict=prepped_obs, reference_dict=preprocessor.observation_space.original_space)
        return preprocessor.transform(prepped_obs)


def reorder_observations(observations_dict: dict, reference_dict: dict) -> dict:
    """Reorders the observations according to some reference dictionary

    Parameters
    ----------
    observations_dict : dict
        Dictionary of observations
    reference_dict : dict
        Reference dict

    Returns
    -------
    dict
        Reordered observations
    """
    reordered_obs = OrderedDict()
    for key in reference_dict:
        reordered_obs[key] = observations_dict[key]
    return reordered_obs
