"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Transforms the raw observations from the ACT3MultiAgentEnv into the transformed
format a policy expects.
"""

import copy
import dataclasses
import typing
from collections import OrderedDict
from pathlib import Path

import gymnasium
import numpy as np
import torch
from flatten_dict import flatten_dict
from pydantic import BaseModel, ConfigDict
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.typing import AgentID

from corl.environment.multi_agent_env import ACT3MultiAgentEnv
from corl.evaluation.episode_artifact import EpisodeArtifact, Preprocessor


@dataclasses.dataclass
class AgentCheckpoint:
    """Agent Checkpoint and Name"""

    agent_name: AgentID
    checkpoint_dir: str | Path


class ACT3MultiAgentEnvToPolicyNetworkValidator(BaseModel):
    """Validator for Network Explainability"""

    agent_checkpoints: list[AgentCheckpoint]
    env_config: dict[str, typing.Any]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ACT3MultiAgentEnvToPolicyNetworkInputs:
    """Given an environment configuration and agent checkpoints,
    takes the necessary steps to transform the raw environment observations
    into the policy network inputs.

    Parameters
    ----------
    agent_checkpoints : typing.List[AgentCheckpoint]
        A list of AgentCheckpoints
    env_config : typing.Union[EnvContext, typing.Dict]
        The environment configuration used to recreate the ACT3MultiAgentEnv
    """

    def __init__(self, **kwargs) -> None:
        self.config: ACT3MultiAgentEnvToPolicyNetworkValidator = self.get_validator()(**kwargs)
        self.env = ACT3MultiAgentEnv(self.config.env_config)
        self.policies: dict[AgentID, PPOTorchPolicy] = load_checkpoints(self.config.agent_checkpoints)
        self._policy_input_observation_names: list[str] = []
        self.discrete_inputs_indicator: np.ndarray = np.array([])

    @staticmethod
    def get_validator() -> type[ACT3MultiAgentEnvToPolicyNetworkValidator]:
        """Gets validator for class"""
        return ACT3MultiAgentEnvToPolicyNetworkValidator

    @property
    def policy_input_observation_names(self):
        """A list of strings indicating each network input"""
        return self._policy_input_observation_names

    @policy_input_observation_names.setter
    def policy_input_observation_names(self, value):
        """Custom setter that checks to ensure the expected policy inputs have not
        changed between episode artifacts"""
        if self._policy_input_observation_names:
            if self._policy_input_observation_names != value:
                raise ValueError(
                    "Network outputs changed between episoide artifacts.", f" {self._policy_input_observation_names} -> {value}"
                )
        else:
            self._policy_input_observation_names = value

    def _get_preprocessor_and_filter(
        self,  # noqa: PLR6301
        episode_artifact: EpisodeArtifact,
    ) -> tuple[dict[AgentID, Preprocessor], dict[AgentID, Filter]]:
        """Returns the policy preprocessor and filter. When `sanity_check` is True, will loop through
        preprocessors and filters to ensure they are the same throughout the episode artifact"""
        policy_preprocessor: dict[AgentID, Preprocessor] = {}
        policiy_filter: dict[AgentID, Filter] = {}
        for policy_id, policy_artifact in episode_artifact.policy_artifact.items():
            policy_preprocessor[policy_id] = policy_artifact.preprocessor
            policiy_filter[policy_id] = policy_artifact.filters
        return policy_preprocessor, policiy_filter

    def get_raw_observations(
        self, agent_id: AgentID, episode_artifact: EpisodeArtifact, include_initial_state: bool = True  # noqa: PLR6301
    ) -> list[OrderedDict[typing.Any, typing.Any]]:
        """Returns the raw observations for a policy from the episode artifact

        Parameters
        ----------
        agent_id : AgentID
            The agent id to grab the observations for.
        episode_artifact : EpisodeArtifact
            The episode artifact to get the observations from.
        include_initial_state : bool
            When True, includes the initial state of the episode. By default, True.

        Returns
        -------
        typing.List[typing.Dict[str, typing.Any]]
            A list of raw observations.
        """
        raw_observations: list[OrderedDict] = []

        # The initial state is not available from the simulator steps
        # and, therefore, we (optionally) append it here.
        if include_initial_state:
            agent_initial_state = episode_artifact.initial_state[agent_id]
            raw_observations.append(agent_initial_state)
        # Grabs the names of the observations that the policy network expects

        for step in episode_artifact.steps:
            agent_step: EpisodeArtifact.AgentStep = step.agents[agent_id]
            agent_obs: OrderedDict[typing.Any, typing.Any] = agent_step.observations
            # Removes any observations that were present in the environment and excluded from training
            raw_observations.append(agent_obs)
        return raw_observations

    def _get_policy_input_observation_names(self, input_obs: np.ndarray, preprocessor: Preprocessor) -> list[str]:
        """Returns the policy input observation names and sets `_policy_input_observation_names` and
        `discrete_inputs_indicator`.

        Parameters
        ----------
        agent_id : AgentID
            The Agent ID corresponding to the policy to use
        input_obs : np.ndarray
            An array representing an example policy input sample.

        Returns
        -------
        typing.List[str]
            A list of policy input observation names
        """
        assert len(input_obs.shape) == 1, "The `input_obs` is assumed to be a flattened array of shape num_input_features."
        # NOTE: The order of the observation space as its listed in the OrderedDict matters. We use the one that is an attribute
        # of the preprocessor itself, since the preprocessor handles the transformation. We have had issues with the observation space
        # found in the loaded checkpoint are ordered differently from the observation space found in the preprocessor.

        # This will output a dict of arrays (pre-flattened arrays)
        _original_space_dict = preprocessor.observation_space.original_space
        ref_idx_dict = restore_original_dimensions(
            np.expand_dims(np.arange(0, len(input_obs)), axis=0), obs_space=preprocessor.observation_space, tensorlib=np
        )
        spaces_dict_restored_dims = restore_original_dimensions(
            np.expand_dims(input_obs, axis=0), obs_space=preprocessor.observation_space, tensorlib=np
        )
        # Flattens various dictionaries that keep track of spaces and indices
        flattened_spaces_dict = _to_flattened_dict(input_dict=spaces_dict_restored_dims)
        flattened_original_spaces_dict = _to_flattened_dict(input_dict=_original_space_dict)
        ref_idx_dict = _to_flattened_dict(input_dict=ref_idx_dict)

        policy_input_names = [""] * len(input_obs)
        discrete_value_indicator = 1
        discrete_values = np.zeros_like(input_obs)
        for key in flattened_spaces_dict:
            obs_tuple = tuple(name for name in _flatten_tuples(key) if name != "direct_observation")
            obs_name = ".".join(obs_tuple)
            space_ = flattened_original_spaces_dict[key]
            _idx_array = ref_idx_dict[key].flatten()
            # When the space is either Discrete or MultiDiscrete, add a non-zero value to the
            # discrete inputs indicator. All one-hot encoded vectors representing the same feature are represented
            # by the same int value. This is used later to consolidate the one-hot encoded vectors for visualization
            if isinstance(space_, gymnasium.spaces.MultiDiscrete | gymnasium.spaces.Discrete):
                discrete_values[_idx_array] = discrete_value_indicator
                discrete_value_indicator += 1
            for grouped_idx, _idx in enumerate(_idx_array):
                policy_input_names[_idx] = f"{obs_name}_{grouped_idx}"
        self._policy_input_observation_names = policy_input_names
        self.discrete_inputs_indicator = discrete_values
        return policy_input_names

    def _get_transformed_obs(
        self,  # noqa: PLR6301
        prepped_obs: dict[str, typing.Any],
        preprocessor: Preprocessor,
        policy_filter: typing.Any,
    ) -> np.ndarray:
        """Given a raw observation and a policy-id, transforms the raw observations into a numpy array.

        Parameters
        ----------
        prepped_obs : typing.Dict[str, typing.Any]
            The observation prepared for training
        preprocessor : Preprocessor
            The policy preprocessor
        filter : Filter
            The policy filter(s)

        Returns
        -------
        np.ndarray
            The transformed observations.
        """
        if preprocessor is not None:
            transformed_obs = preprocessor.transform(prepped_obs)
        return policy_filter(transformed_obs)

    @staticmethod
    def _del_keys(raw_obs: dict[str, typing.Any], remove_keys: set):
        for key in remove_keys:
            del raw_obs[key]
        return raw_obs

    def raw_observations_to_sample_batch(
        self,
        agent_id: AgentID,
        episode_artifact: EpisodeArtifact,
        include_initial_state: bool = True,
    ) -> SampleBatch:
        """Converts the raw observations to the expected policy network input format.
        This includes any normalizations or transformations that are configured to happen by
        the environment.

        Parameters
        ----------
        agent_id : AgentID
            A string indicating the agent-id
        episode_artifact: EpisodeArtifact
            Episode artifact to process
        include_initial_state: bool
            When True, includes the initial state (timestep 0).

        Returns
        -------
        SampleBatch
            Sample batch of observations.
        """
        batch_obs = []
        raw_observations = copy.deepcopy(
            self.get_raw_observations(agent_id=agent_id, episode_artifact=episode_artifact, include_initial_state=include_initial_state)
        )
        policy_artifact: EpisodeArtifact.PolicyArtifact = episode_artifact.policy_artifact[agent_id]
        _agent = self.env.agent_dict[agent_id]
        for idx, obs in enumerate(raw_observations):
            array_obs = self._get_transformed_obs(
                _agent.create_training_observations(obs),
                preprocessor=get_preprocessor(self.env.observation_space[agent_id])(self.env.observation_space[agent_id]),  # type: ignore
                policy_filter=policy_artifact.filters,
            )
            batch_obs.append(array_obs)
            if idx == 0:
                self._get_policy_input_observation_names(
                    input_obs=array_obs,
                    preprocessor=get_preprocessor(self.env.observation_space[agent_id])(  # type: ignore
                        self.env.observation_space[agent_id]  # type: ignore
                    ),
                )
        view_requirements = self.policies[agent_id].model.view_requirements
        sample_batch_list = []

        collector = AgentCollector(view_reqs=view_requirements, max_seq_len=1, is_training=False)
        for idx, one_batch in enumerate(batch_obs):
            # Adds an observation to the buffer, the rest of the inputs are dummy variables
            collector.add_init_obs(episode_id=0, agent_index=0, env_id=0, init_obs=one_batch, t=idx)
            sample_batch_list.append(collector.build_for_inference())
        final_sample_batch = concat_samples(sample_batch_list)
        return SampleBatch({key: torch.from_numpy(array_value) for key, array_value in final_sample_batch.items()})


def _flatten_tuples(input_tuple: tuple):
    for value in input_tuple:
        if isinstance(value, tuple):
            yield from _flatten_tuples(value)
        else:
            yield value


def _to_flattened_dict(input_dict: dict[typing.Any, typing.Any], reducer: str = "tuple"):
    """Helper function to turn all nested dicts and repeated values into flattened dictionaries -
    when no repeated values are found, the input dict will still be flattened"""
    while any(isinstance(_value, dict | gymnasium.spaces.Dict | RepeatedValues | Repeated) for _, _value in input_dict.items()):
        new_dict = {}
        for key, value in input_dict.items():
            if isinstance(key, tuple):
                key = tuple(_flatten_tuples(key))  # noqa: PLW2901
            if isinstance(value, RepeatedValues):
                new_dict[key] = value.values
            elif isinstance(value, Repeated):
                new_dict[key] = value.child_space
            else:
                new_dict[key] = value
        input_dict = flatten_dict.flatten(new_dict, reducer=reducer)
    return flatten_dict.flatten(input_dict, reducer=reducer)


def load_checkpoints(checkpoints: list[AgentCheckpoint]) -> dict[AgentID, PPOTorchPolicy]:
    """Loads a checkpoint

    Parameters
    ----------
    checkpoints : typing.List[AgentCheckpoint]
        A list of agent checkpoints.

    Returns
    -------
    typing.Dict[AgentID, Policy]
        Loaded checkpoints
    """
    # By default, the checkpoint is loaded onto the device it was trained on.
    # To avoid mismatches between the model and the tensors later on, we load
    # the model onto the CPU.
    DEVICE = "cpu"
    loaded_checkpoints = {}
    for agent_checkpoint in checkpoints:
        _policy = Policy.from_checkpoint(agent_checkpoint.checkpoint_dir)
        # Set the device
        if isinstance(_policy, Policy):
            _policy.devices = [DEVICE]
            _policy.device = DEVICE

            _policy.model = _policy.model.to(DEVICE)
            loaded_checkpoints[agent_checkpoint.agent_name] = _policy
        else:
            _policy[agent_checkpoint.agent_name].devices = [DEVICE]
            _policy[agent_checkpoint.agent_name].device = DEVICE

            _policy[agent_checkpoint.agent_name].model = _policy[agent_checkpoint.agent_name].model.to(DEVICE)
            loaded_checkpoints[agent_checkpoint.agent_name] = _policy[agent_checkpoint.agent_name]
    return loaded_checkpoints
