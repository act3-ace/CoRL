"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Network Explainability Tools
"""

import typing
from warnings import warn

import gymnasium
import numpy as np
import torch
from flatten_dict import flatten_dict
from pydantic import BaseModel, ConfigDict
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation import postprocessing as ray_postprocessing
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.visualization.network_explainability.env_policy_transforms import ACT3MultiAgentEnvToPolicyNetworkInputs, AgentCheckpoint

VALUE_FUNCTION_PRED = "value_pred"
SAMPLE_BASED_FEAT_IMPORTANCE = "sample_based"
GRAD_BASED_FEAT_IMPORTANCE = "gradient_based"


class NetworkExplainabilityValidator(BaseModel):
    """Validator for Network Explainability"""

    value_function_prediction: str = VALUE_FUNCTION_PRED
    network_output_per_controller: tuple[str, str] = ("mean", "std")
    ignore_inputs_containing: str = "invalid"
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PPOTorchNetworkExplainability:
    """Generates feature importance metrics and plots."""

    def __init__(
        self,
        agent_checkpoints: list[AgentCheckpoint],
        env_config: dict[str, typing.Any],
        value_function_prediction: str = VALUE_FUNCTION_PRED,
        network_output_controller: tuple[str, str] = ("mean", "std"),
    ) -> None:
        """
        Initializes the policies and environment for a set of trajectories. Each trajectory contained
        in an EpisodeArtifact is expected to be passed into each feature importance method.

        Parameters
        ----------
        agent_checkpoints : typing.List[AgentCheckpoint]
            A list of AgentCheckpoints
        env_config : typing.Union[EnvContext, typing.Dict]
            The environment configuration used to recreate the ACT3MultiAgentEnv
        value_function_prediction : typing.Optional[str], optional
            The name of the value function prediction, by default VALUE_FUNCTION_PRED
        network_output_per_controller : typing.Tuple[str], optional
            This should indicate what the policy networkout outputs. For example, for PPO, the outputs
            are the mean and standard deviation of each controller, by default ('mean', 'std').
        """

        self.config: NetworkExplainabilityValidator = self.get_validator()(
            value_function_prediction=value_function_prediction, network_output_per_controller=network_output_controller
        )

        self.env_policy_input_transformer = ACT3MultiAgentEnvToPolicyNetworkInputs(
            agent_checkpoints=agent_checkpoints, env_config=env_config
        )

        self.feat_importance_gradient_based: dict[AgentID, list[dict[str, typing.Any]]] = self._initialize_feature_dict()
        self.feat_importance_sample_based: dict[AgentID, list[dict[str, typing.Any]]] = self._initialize_feature_dict()
        self.vf_residuals: dict[AgentID, list[typing.Any]] = self._initialize_feature_dict()
        self._network_output_names: list[str] = []

    def _initialize_feature_dict(self) -> dict[AgentID, list[dict[str, typing.Any]]]:
        """Helper function that returns an empty dictionary with the agent-ids as the keys"""

        return {agent_id: [] for agent_id, _ in self.env_policy_input_transformer.policies.items()}

    @staticmethod
    def get_validator() -> type[NetworkExplainabilityValidator]:
        """Gets validator for class"""
        return NetworkExplainabilityValidator

    @property
    def network_output_names(self):
        """A list of strings indicating each network output"""
        return self._network_output_names

    @network_output_names.setter
    def network_output_names(self, value):
        """Custom setter that checks to ensure the value of the outputs have not
        changed between episode artifacts"""
        if self._network_output_names:
            if self._network_output_names != value:
                raise ValueError("Network outputs changed between episoide artifacts.", f" {self._network_output_names} -> {value}")
        else:
            self._network_output_names = value

    def generate_vf_residuals(self, episode_artifact: EpisodeArtifact, agent_id: AgentID | None = None) -> dict[AgentID, np.ndarray]:
        """
        Generates value function residuals for the provided episode,
        and returns the residuals in a dictionary of
        [key: agent_id, values: policy_residuals].

        See https://github.com/ray-project/ray/blob/ray-2.2.0/rllib/evaluation/sampler.py#L756
        for how the postprocessed_batch is generated. The postprocessed_batch, accessible
        in the postprocessed_trajectory method of the default callback, contains the
        value predictions and value targets. This method aims to replicate the logic.

        Parameters
        ----------
        episode_artifact : EpisodeArtifact
            An episode artifact to pull the trajectory from.
        agent_id : typing.Optional[AgentID], optional
            The agent-id when None, the feature importance for all policies will
            be generated, by default None

        Returns
        -------
        typing.Dict[AgentID, np.ndarray]
            A dictionary where the top-level key is the agent id and
            the values are a numpy array of the residuals.
        """

        process_policies: list = list(self.env_policy_input_transformer.policies.keys())
        # If an agent is specified, overwrite the list of policies to process
        if agent_id:
            process_policies = [agent_id]

        vf_residuals_per_policy = {}
        for _agent_id in process_policies:
            # Get policy for current agent
            _policy = self.env_policy_input_transformer.policies[_agent_id]

            # Get a temporary instance of the sample_batch to extract
            # value function predictions
            sample_batch = self.env_policy_input_transformer.raw_observations_to_sample_batch(
                _agent_id, episode_artifact, include_initial_state=True
            )

            # Get value function predictions
            _, _, info = _policy.compute_actions_from_input_dict(sample_batch)
            vf_preds = info["vf_preds"]

            # Get rewards from episode artifact steps: Note this does not include the initial state.
            # These are the rewards received at a given timestep
            rewards = np.array([s.agents[_agent_id].rewards for s in episode_artifact.steps])
            rewards = np.array([np.array(list(reward.values())).sum() for reward in rewards])
            rewards = np.array(rewards)
            # Get dones: does not include the initial state
            # --> rllib SampleBatch expects truncated and terminated conditions not dones
            terminateds = np.zeros_like(rewards)
            truncateds = np.zeros_like(rewards)
            # Check if the episode completed with a done condition.
            # If so, add a True (=1) to the end of the done condition
            # array (assuming here that the done is the last step in
            # the episode).

            _agent_platforms = episode_artifact.agent_to_platforms[_agent_id]
            for _agent_platform in _agent_platforms:
                if any(episode_artifact.dones[_agent_platform].values()):
                    terminateds[-1] = 1
                    truncateds[-1] = 1
                    # If the episode ends in a done condition, remove the value
                    # prediction of the terminal state. It will be replaced by a reward
                    # of 0 when `ray_postprocessing.compute_gae_for_sample_batch` is called
                    vf_preds = vf_preds[:-1]

            # Add rewards, dones, and vf_preds to new sample batch
            # We create a new SampleBatch instance because the
            # previous SampleBatch causes multiple issues due to
            # device-mismatches. This approach is simpler than using
            # the already-defined sample_batch and running the list of
            # required device-swapping functions.
            sample_batch_new = SampleBatch(
                {
                    SampleBatch.REWARDS: rewards,
                    SampleBatch.TERMINATEDS: terminateds,
                    SampleBatch.TRUNCATEDS: truncateds,
                    SampleBatch.VF_PREDS: vf_preds,
                }
            )

            # Get post-processed sample batch from which we can extract
            # the value function targets.
            postprocessed_sample_batch = ray_postprocessing.compute_gae_for_sample_batch(_policy, sample_batch_new)
            vf_targets = postprocessed_sample_batch["value_targets"]
            vf_residuals = vf_targets - vf_preds
            vf_residuals_per_policy[_agent_id] = vf_residuals

        return vf_residuals_per_policy

    @property
    def network_input_names(self):
        """Network input names"""
        return self.env_policy_input_transformer.policy_input_observation_names

    def generate_feat_importance_grad_based(
        self,
        episode_artifact: EpisodeArtifact,
        agent_id: AgentID | None = None,
        policy_network_input_sample_batch_key_map: dict | None = None,
    ) -> dict[AgentID, typing.Any]:
        """Generates gradient-based feature importance for one episode, returns the feature
        importance in a dictionary.

        The features are generated for each observation and output, where the output
        for each policy is shaped: [episode_length, number of observations], where
        the name of each observation is stored in `self.env_policy_transforms.policy_input_observation_names`

        Parameters
        ----------
        episode_artifact : EpisodeArtifact
            An episode artifact to pull the trajectory from.
        agent_id : typing.Optional[AgentID], optional
            The policy-id, when None, the feature importance for all policies will
            be generated, by default None
        policy_network_input_sample_batch_key_map : dict, optional
            A dictionary that maps the policy-id to the expected policy network input SampleBatch key.
            When one is not provided, it assumes the policy network input for all policies is `SampleBatch.OBS`.
            This is needed because in order to generate the feature importance, torch.autograd.grad is manually called.

        Returns
        -------
        typing.Dict[AgentID, typing.Any]
            A dictionary where the top-level key is the policy-id and the values are
            a dictionary of gradient-based feature importances.
        """
        process_policies: list = list(self.env_policy_input_transformer.policies.keys())
        # If an agent is specified, overwrite the list of policies to process
        if agent_id:
            process_policies = [agent_id]
        feat_importance_per_policy = {}

        for _agent_id in process_policies:
            # Policies can have different inputs
            input_key = (
                policy_network_input_sample_batch_key_map.get(_agent_id, SampleBatch.OBS)
                if policy_network_input_sample_batch_key_map
                else SampleBatch.OBS
            )
            sample_batch = self.env_policy_input_transformer.raw_observations_to_sample_batch(_agent_id, episode_artifact)
            gradients_dict = self._get_policy_network_gradients(_agent_id, sample_batch, input_key)
            feat_importance_per_policy[_agent_id] = gradients_dict
        return feat_importance_per_policy

    def append_vf_residuals(self, episode_artifact: EpisodeArtifact) -> None:
        """
        Generates the residuals of the value function for the specified
        episode artifact (for all policies, retrieved as
        `self.policies.keys()`) and appends them to the list of
        residuals for the corresponding policiy_id in
        `self.vf_residuals`.

        Parameters
        ----------
        episode_artifact : EpisodeArtifact
            An episode artifact.

        Returns
        -------
        None
        """

        residuals = self.generate_vf_residuals(episode_artifact)
        for agent_id, residual_values in residuals.items():
            self.vf_residuals[agent_id].append(residual_values)

    def append_feat_importance_gradient_based(
        self, episode_artifact: EpisodeArtifact, policy_network_input_sample_batch_key_map: dict | None = None
    ) -> None:
        """Generates the gradient-based feature importance and
        appends it to `self.feat_importance_gradient_based`.

        Parameters
        ----------
        episode_artifact : EpisodeArtifact
            An episode artifact.
        """
        grad_feat_importance = self.generate_feat_importance_grad_based(
            episode_artifact, policy_network_input_sample_batch_key_map=policy_network_input_sample_batch_key_map
        )
        for agent_id, feat_importance in grad_feat_importance.items():
            self.feat_importance_gradient_based[agent_id].append(feat_importance)

    def get_agg_feat_importance_grad_based(
        self, agent_id: AgentID = None, normalize: bool = True, agg_func: typing.Callable = np.mean, test_case: int | None = None
    ) -> dict[AgentID, dict[str, typing.Any]]:
        """Aggregates the gradient-based feature importance according to the `agg_func` across all observations
        and test cases (defined by `test_case`). For example, when `agg_func=np.mean`, returns a dictionary
        that specifies how important each input is to the output, on average across test cases. Assumes the
        `self.append_feat_importance_gradient_based` has already been called to fill out the array of gradient-based feature importances

        Parameters
        ----------
        agent_id : AgentID, optional
            When provided, only returns the aggregated feature importance for the specified agent. Otherwise, all
            agents are returned, by default None
        normalize : bool, optional
            When True, normalizes the aggregated feature importance so that all features for a given
            output sum up to 1 , by default True
        agg_func : typing.Callable, optional
            Specifies how the features should be aggregated, by default np.mean. The callable must take `axis` as an input.
        test_case : typing.Optional[int], optional
            An integer that represents which test cases to process. When none is specified, processes all
            test cases, by default None. Note the test case is the index of `self.feat_importance_gradient_based`

        Returns
        -------
        typing.Dict[AgentID, typing.Dict[str, typing.Any]]
            A nested dictionary that specifies how important each input is to the output.
        """

        feat_importance = self.feat_importance_gradient_based
        policy_input_observation_names = np.array(self.network_input_names.copy())
        discrete_groups = self.env_policy_input_transformer.discrete_inputs_indicator.copy()
        ignore_columns_idx = self._ignore_policy_input_columns()
        discrete_groups = np.delete(discrete_groups, ignore_columns_idx)
        policy_input_observation_names = np.delete(policy_input_observation_names, ignore_columns_idx)

        # Select policies to process, when None, all policies will be processed.
        policies = [agent_id] if agent_id else list(feat_importance.keys())
        agg_feats: dict[AgentID, dict[str, typing.Any]] = {}

        for _agent_id in policies:
            agg_feats[_agent_id] = {}
            policy_feat_importance = feat_importance[_agent_id]
            assert (
                len(policy_feat_importance) > 0
            ), f"No episode artifacts were processed for {_agent_id}, did you forget to call `append_feat_importance_gradient_based`?"
            agg_feat_per_policy = {}
            # When only one test is specified, grab the test case
            if test_case:
                policy_feat_importance = [policy_feat_importance[test_case]]
            # Iterate over test cases and vertically stack the array of features
            for policy_feat_importance_per_test_case in policy_feat_importance:
                for output_feat_name, feat_array in policy_feat_importance_per_test_case.items():
                    if output_feat_name not in agg_feat_per_policy:
                        agg_feat_per_policy[output_feat_name] = feat_array
                    else:
                        agg_feat_per_policy[output_feat_name] = np.vstack((agg_feat_per_policy[output_feat_name], feat_array))

            # Iterate over the one-hot encoded vector groups and store which columns need to be removed
            unique_groups = np.unique(discrete_groups)
            # 0 is the fill value and does not indicate a group of one-hot encoded values.
            unique_groups = unique_groups[unique_groups.nonzero()]

            remove_columns = np.array([]).astype(int)
            grouped_policy_input_observation_names = policy_input_observation_names.copy()
            for group in unique_groups:
                # The [0] refers to the first element of the tuple. Since
                # discrete_groups is a flattened array, there is only one
                # element in the tuple.
                group_idx = np.where(discrete_groups == group)[0]
                # All one-hot encoded vectors are summed together and stored where the first one-hot encoded
                # vector of the group appears
                # Flag the rest of the one-hot encoded vectors for removal
                remove_columns = np.hstack((remove_columns, group_idx[1:]))
                # Consolidate the one-hot encoded vector names: this takes the first one-hot encoded vector, and replaces the
                # suffix (_0)
                grouped_policy_input_observation_names[group_idx[0]] = policy_input_observation_names[group_idx[0]].replace("_0", "")
            # Updates the input observation names accordingly
            grouped_policy_input_observation_names = np.delete(grouped_policy_input_observation_names, remove_columns)

            # Aggregate the feature importance for each policy, we take the absolute value before the aggregate so that
            # the sign of gradient does not impact the metric (e.g. the direction of the gradient is ignored.)
            for output_feat_name, feat_array in agg_feat_per_policy.items():
                feat_array = np.abs(feat_array)  # noqa: PLW2901
                assert len(discrete_groups) == feat_array.shape[-1]
                # Iterate over the one-hot encoded vectors and aggregate them
                for group in unique_groups:
                    group_idx = np.where(discrete_groups == group)[0]
                    feat_array[:, group_idx[0]] = feat_array[:, group_idx].sum(axis=1)
                feat_array = np.delete(feat_array, remove_columns, axis=1)  # noqa: PLW2901
                # NOTE: Each column represents a different input and each row
                # represents a different observation
                _aggregated = agg_func(feat_array, axis=0)
                # Normalize so that each feature importance adds up to 1
                if normalize:
                    _aggregated = _aggregated / _aggregated.sum()

                assert len(_aggregated) == len(grouped_policy_input_observation_names)
                agg_feats[_agent_id].update({output_feat_name: dict(zip(grouped_policy_input_observation_names, _aggregated))})

        return agg_feats

    def _get_policy_network_gradients(
        self, agent_id: AgentID, sample_batch: SampleBatch, policy_network_input_sample_batch_key: str = SampleBatch.OBS
    ) -> dict[str, np.ndarray]:
        """Returns the gradient of each input with respect to the output feature

        Parameters
        ----------
        agent_id : AgentID
            The agent id.
        sample_batch : SampleBatch
            The inputs to the policy network
        policy_network_input_sample_batch_key : str, optional.
            The key in SampoleBatch that corresponds to the input of the policy network. By default,
            `SampleBatch.OBS`.

        Returns
        -------
        Dict[str, np.ndarray]
            Each array represents the gradient of the input features with respect
            one output feature, where the output feature is the name of the key. For example,
            when the key is the mean of the acceration, the array represents the gradients of all
            observations with respect to the mean acceelration. Each array is shaped: [batch_size, num_input_feats]
        """
        policy_network: PPOTorchPolicy = self.env_policy_input_transformer.policies[agent_id]

        # requires_grad must be set to true in order to compute the gradient below.
        sample_batch[policy_network_input_sample_batch_key].requires_grad = True

        # shape: [batch_size, num_output_feats]
        output_tensor, _ = policy_network.model(sample_batch)
        _, num_output_feats = output_tensor.shape
        value_preds_tensor = policy_network.model.value_function()

        ignore_idx = self._ignore_policy_input_columns()
        expected_num_columns = len(self.network_input_names)

        # The network outputs are ordered accordingly
        for idx in range(num_output_feats):
            # Grabs one output feature (across all samples)
            output_feature = output_tensor[:, idx]
            # Retain the graph so that we can compute gradients wrt all outputs. By default,
            # the graph used to compute the gradient will be freed after the grad call.
            # Shape: [batch_size, num_input_feats]
            grad_tensor = torch.autograd.grad(
                outputs=output_feature.sum(), inputs=sample_batch[policy_network_input_sample_batch_key], retain_graph=True
            )
            # The grad_tensor is a tuple of length 1
            grad_array = grad_tensor[0].detach().numpy()
            # Sanity check on the number of columns
            assert grad_array.shape[-1] == expected_num_columns
            grad_array = self._check_and_process_extra_inputs(
                view_requirements=policy_network.model.view_requirements,
                grad_array=grad_array,
                policy_network_input_sample_batch_key=policy_network_input_sample_batch_key,
            )

            # These columns don't correspond to input features and are used for padding
            if ignore_idx:
                grad_array = np.delete(grad_array, ignore_idx, axis=1)
            # Stack arrays in sequence depth wise (in the third dimension)
            # we can ignore F821 because gradients is defined the first iteration
            gradients: np.ndarray = grad_array if idx == 0 else np.dstack((gradients, grad_array))  # noqa: F821

        assert isinstance(policy_network.action_space, gymnasium.spaces.Dict)
        output_gradients_dict, network_output_names = self._process_outputs(
            action_space=policy_network.action_space, output_features=gradients, num_output_feats=num_output_feats
        )
        # Compute the gradient for the value function output
        grad_tensor = torch.autograd.grad(outputs=value_preds_tensor.sum(), inputs=sample_batch[policy_network_input_sample_batch_key])
        grad_array = grad_tensor[0].detach().numpy()
        grad_array = self._check_and_process_extra_inputs(
            view_requirements=policy_network.model.view_requirements,
            grad_array=grad_array,
            policy_network_input_sample_batch_key=policy_network_input_sample_batch_key,
        )
        if ignore_idx:
            grad_array = np.delete(grad_array, ignore_idx, axis=1)
        output_gradients_dict[self.config.value_function_prediction] = grad_array
        network_output_names.append(self.config.value_function_prediction)
        self.network_output_names = network_output_names
        return output_gradients_dict

    def _check_and_process_extra_inputs(
        self,  # noqa: PLR6301
        view_requirements: dict,
        grad_array: np.ndarray,
        policy_network_input_sample_batch_key: str,
    ):
        """
        When `policy_network_input_sample_batch_key` is not the `SampleBatch.OBS` and the gradient array is 3-dimensional, it's assumed
        the dimensions correspond to [batch_size, sequence_length, num_inputs]. This function (1) checks to ensure that the  policy network
        inputs are derived from `SampleBatch.OBS` and (2) the second dimension of the gradient array corresponds to the length of the
        model view's `shift_arr` and (3) sums the absolute value of the gradient array across the sequence axis.
        """
        if policy_network_input_sample_batch_key != SampleBatch.OBS and len(grad_array.shape) > 2:
            view_reqs = view_requirements[policy_network_input_sample_batch_key]
            assert view_reqs.data_col == SampleBatch.OBS, "Only policy inputs based on the observation are permitted."
            assert grad_array.shape[1] == len(
                view_reqs.shift_arr
            ), """Dimension mismatch:
                the second dimension is assumed to be the the number of observation states that the input depends on, given by the
                length of `ViewRequirements.shift_arr`."""
            grad_array = np.abs(grad_array)
            grad_array = grad_array.sum(axis=1)
            warn(
                f"""The absolute value of the gradient-based feature importance was \
                    aggregated across the {policy_network_input_sample_batch_key} axis. \
                    That is, each feature importance is the |gradient| of the sum of the previous
                    {view_reqs.shift} inputs with respect to each output."""
            )
        return grad_array

    def _process_outputs(
        self, action_space: gymnasium.spaces.Dict, output_features: np.ndarray, num_output_feats: int
    ) -> tuple[dict[str, typing.Any], list[str]]:
        """Helper function to process the output features. Checks the action space type and aggregates
        any Discrete or MultiDiscrete one-hot encoded vectors

        Parameters
        ----------
        action_space : typing.Dict[str, typing.Any]
            Action space
        output_features : np.ndarray
            Output features to check. Expected shape is [batch_size, num_input_features, num_output_features]
        num_output_feats : int
            Number of expected output features

        Returns
        -------
        typing.Dict[str, typing.Any]
            A dictionary of non-aggregated feature importances
        typing.List[str]
            A list of network output names.

        Raises
        ------
        NotImplementedError
            When a Discrete space starts at a value > 0
        NotImplementedError
            For any gymnasium spaces other than MultiDiscrete, Discrete and Box.
        """
        network_output_names = []
        output_feats_dict: dict[str, typing.Any] = {}
        assert isinstance(action_space, gymnasium.spaces.Dict)
        flattened_action_space = flatten_dict.flatten(action_space, reducer="dot")
        # gradients shape: [batch_size, num_input_feats, num_output_feats]
        for action_space_name, _gymnasium_space in flattened_action_space.items():
            # When the gymnasium space is multidiscrete, we aggregate the gradients of the one-hot encoded vectors.
            if isinstance(_gymnasium_space, gymnasium.spaces.MultiDiscrete | gymnasium.spaces.Discrete):
                # This is a mapping of the index to each action space
                ref_idx_dict = restore_original_dimensions(
                    np.expand_dims(np.arange(0, num_output_feats), axis=0), obs_space=action_space, tensorlib=np
                )
                ref_idx_dict = flatten_dict.flatten(ref_idx_dict, reducer="dot")
                # Grabs the index (action space output) corresponding to each action
                idx_range = ref_idx_dict[action_space_name].flatten()
                # Grab the outputs corresponding to this space
                gradients_subset = output_features[:, :, idx_range].copy()

                class_name = _gymnasium_space.__class__.__name__
                start = 0
                if isinstance(_gymnasium_space, gymnasium.spaces.MultiDiscrete):
                    nvecs = _gymnasium_space.nvec
                if isinstance(_gymnasium_space, gymnasium.spaces.Discrete):
                    if _gymnasium_space.start > 0:
                        raise NotImplementedError
                    nvecs = np.array([_gymnasium_space.n])
                else:
                    raise RuntimeError(f"gymnasium_space must be MultiDiscrete or Discrete, got {_gymnasium_space}")
                # The one hot encoding is described here: https://docs.ray.io/en/latest/rllib/rllib-models.html#built-in-preprocessors
                for one_hot_vec_span in nvecs:
                    # sum across the depth dimension
                    # gradients_subset shaped: batch_size, num_input_features, num_output_features]
                    # gradients_agg shaped: [batch_size, num_input_features] - sum is taken across depth
                    gradients_agg = gradients_subset[:, :, start : start + one_hot_vec_span].sum(axis=2)
                    # The output name, e.g. controller_MultiDiscrete[2]
                    output_name = f"{action_space_name}_{class_name}[{one_hot_vec_span}]"
                    network_output_names.append(output_name)
                    output_feats_dict[output_name] = gradients_agg
                    # The start of the next group will be where this group ended
                    start = one_hot_vec_span

            # If its a box, the outputs are the mean and std deviation.
            elif isinstance(_gymnasium_space, gymnasium.spaces.Box):
                _box_network_output_names = self._get_network_output_names(action_space)

                assert output_features.shape[2] == len(_box_network_output_names)
                for _idx, name in enumerate(_box_network_output_names):
                    output_feats_dict[name] = output_features[:, :, _idx]
                    network_output_names.append(name)
            # If its any other type of space, raise a NotImplemetnedError
            else:
                raise NotImplementedError("Only implemented for gymnasium spaces: MultiDiscrete, Discrete and Box.")

        return output_feats_dict, network_output_names

    def _ignore_policy_input_columns(self):
        """Helper function to identify columns that either correspond to a pad column (relevant for repeated obs)
        or a column with the name matching `ignore_inputs_containg` (e.g. invalid)"""
        return [
            idx
            for idx, name in enumerate(self.network_input_names)
            if (name == "") or (self.config.ignore_inputs_containing in name)  # noqa: PLC1901
        ]

    def _filtered_policy_input_columns(self):
        return [
            name
            for name in self.network_input_names
            if (name != "") and (self.config.ignore_inputs_containing not in name)  # noqa: PLC1901
        ]

    def _get_network_output_names(self, action_space_struct: gymnasium.spaces.Dict) -> list[str]:
        """Given a dict of actions, returns a list of network outputs"""
        # Returns a flattened dictionary, where all keys are separated by a dot.
        action_space_struct = flatten_dict.flatten(action_space_struct, reducer="dot")
        action_space_keys = list(action_space_struct.keys())
        network_outputs = []
        for action_name in action_space_keys:
            network_outputs.extend([f"{name}_{action_name}" for name in self.config.network_output_per_controller])
        return network_outputs
