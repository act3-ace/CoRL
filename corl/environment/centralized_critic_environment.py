"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Extra callbacks that can be for evaluating during training to
generate episode artifacts.
"""

import typing
import warnings
from collections import OrderedDict

import gymnasium
import numpy as np
import tree
from pydantic import ImportString
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from corl.environment.multi_agent_env import ACT3MultiAgentEnv, ACT3MultiAgentEnvValidator
from corl.environment.utils.obs_buffer import ObsBuffer
from corl.libraries.units import Quantity


class CorlCentralizedCriticActionFill(DefaultCallbacks):
    """
    Fills in the opponent actions info in the training batches. The class leverages the action dict sample
    to determine the structure of acts to populate with batch data.
    """

    @staticmethod
    def update_dict_with_values(data_dict, update_list):
        """
        This function updates a dictionary with single values from a list.

        Args:
            data_dict: The dictionary to be updated.
            update_list: A list containing single values to update specific keys in data_dict.

        Returns:
            A new dictionary with the updated structure.
        """
        # Create a copy to avoid modifying the original dictionary during iteration
        updated_dict = data_dict.copy()

        # Enumerate the update list to get index and value
        for index, value in enumerate(update_list):
            # Assuming the order of values in the list corresponds to the dictionary's order
            # Update the value at the current index in the dictionary
            updated_dict[list(updated_dict.keys())[index]] = value

        return updated_dict

    def on_postprocess_trajectory(
        self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs  # noqa: PLR6301
    ):
        """Replaces the actions stored as part of obs which are off by 1 with the actual actions at timestamp"""
        # TODO THIS WILL NEED TO BE UPDATED TO HANDLE MA CASE... Not in setup on CORL currently
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_policy_ids = [policy for policy in policies if policy != policy_id]
        other_ids = [agent for agent in worker.env._agent_ids if agent != policy_id]  # noqa: SLF001
        for other_id in other_ids:
            if other_id in other_policy_ids:
                action_encoder = ModelCatalog.get_preprocessor_for_space(worker.env.action_space[other_id])
                # set the other actions into the observation
                # ? No value for single policy ? TODO
                _, _, other_policy_batch = original_batches[episode.policy_mapping_fn(other_id, episode, worker)]
                # sample actions for populating and encoding
                other_action_sample = worker.env.action_space[other_id].sample()

                if worker.env.config.disable_action_history:
                    other_actions = np.array(
                        [
                            action_encoder.transform(CorlCentralizedCriticActionFill.update_dict_with_values(other_action_sample, a))
                            for a in other_policy_batch[SampleBatch.ACTIONS]
                        ]
                    )
                else:
                    other_actions = np.array([action_encoder.transform(a[other_id]) for a in worker.env._actions])  # noqa: SLF001

                to_update[:, -other_actions[0].size :] = other_actions
            elif worker.env.config.process_nontrainables:
                if worker.env.config.disable_action_history:
                    raise ValueError("The following callback will no work without env.config.disable_action_history set to False")
                action_encoder = ModelCatalog.get_preprocessor_for_space(
                    worker.env.observation_space[policy_id][CorlCentralizedCriticEnvWrapper.OTHER_ACT_STR][other_id]
                )
                other_actions = np.array([action_encoder.transform(a[other_id]) for a in worker.env._actions])  # noqa: SLF001
                to_update[:, -other_actions[0].size :] = other_actions


class CentralCriticObserver:
    def __init__(self, agents, trainable_agents, non_trainable_agents, process_nontrainables: bool):
        self._agents = agents
        self._trainable_agents = trainable_agents
        self._non_trainable_agents = non_trainable_agents
        self._process_nontrainables = process_nontrainables

    def __call__(self, agents_obs, agents_act, observation_space, non_training_obs):
        """
        Rewrites the agent obs to include opponent data for training.
        """
        new_obs = OrderedDict()
        # We are only going to augment trainable items... Same as processing the
        # `self._trainable_agent_dict` keys
        for own_label in agents_obs:
            # Generate the combined oservation space
            [other_label for other_label in agents_obs if own_label != other_label]
            # populate the observation
            new_obs[own_label] = OrderedDict()
            # maintain the current observation
            new_obs[own_label][CorlCentralizedCriticEnvWrapper.OWN_OBS_STR] = agents_obs[own_label]
            # populate the other obs and action ...
            other_obs, other_act = self.get_other_info(own_label, agents_obs, agents_act, observation_space, non_training_obs)
            new_obs[own_label][CorlCentralizedCriticEnvWrapper.OTHER_OBS_STR] = other_obs
            new_obs[own_label][CorlCentralizedCriticEnvWrapper.OTHER_ACT_STR] = other_act

        return new_obs

    def get_other_info(
        self, current_agent_str: str, in_agents_obs, in_agents_act, observation_space, non_training_obs
    ) -> tuple[OrderedDict, OrderedDict]:
        """Gets the observation and action space for other agents

        Args:
            current_agent_str (str): current agent which we are adding obs to

        Returns:
            tuple[OrderedDict, OrderedDict]: tuple containing both observation and action spaces for other agents
        """
        other_obs = OrderedDict()
        other_act = OrderedDict()
        # process all know agents to ensure we get privalaged information for the centralized critic
        for key in self._agents:
            # ignore the key that is currently being augmented
            if key != current_agent_str:
                # Determine if we have action ... If not generate it...
                # ? TODO Single step noise at reset... is this ok ?
                # N.B. These will be replaced with batch updates in callback...
                other_act[key] = (
                    in_agents_act[0][key]
                    if key in in_agents_act[0]
                    else observation_space[current_agent_str][CorlCentralizedCriticEnvWrapper.OTHER_ACT_STR][key].sample()
                )
                # Determine if we have observations ... If not generate it and make sure it is normalized...
                # ? TODO Single step noise at reset... is this ok ?
                if key in in_agents_obs:
                    other_obs[key] = in_agents_obs[key]
                elif self._process_nontrainables:
                    other_obs[key] = non_training_obs[key]

        return other_obs, other_act


class CorlCentralizedCriticEnvWrapperValidator(ACT3MultiAgentEnvValidator):
    """Validation model for the inputs of ACT3MultiAgentEnv"""

    # allow users to supply a observation processing function to meet their needs
    observation_fn: ImportString | None = None
    # Flag for processing non trainables
    # Note this will impact the processing time as it pertains to throughput!!!
    process_nontrainables: bool = True


class CorlCentralizedCriticEnvWrapper(ACT3MultiAgentEnv):
    """
    Multi-agent env that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).
    """

    OWN_OBS_STR = "own_obs"
    OTHER_OBS_STR = "other_obs"
    OTHER_ACT_STR = "other_act"
    OBS_FLAT_STR = "obs_flat"

    def __init__(self, config: dict[str, typing.Any]) -> None:
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
        self.config: CorlCentralizedCriticEnvWrapperValidator
        super().__init__(config)

        #
        # Override the observation space to ensure that we have a format conducive to centralized critic
        #   - Own Observations --- Original Space
        #   - List of Other Obs ---- Observations from all other platforms
        #   - List of Other Actions --- Actions from all other platforms
        #
        # <<N.B.>> The following code utilizesd the full observation space for the other platforms
        # this is to handle the case where the other may be a expert system or some other setup.
        new_observation_space = OrderedDict()

        # We are only going to augment trainable items... Same as processing the
        # `self._trainable_agent_dict` keys
        for agent_k, agent_v in self.observation_space.items():
            # Maintain the existing Own Observation Structure utilized for Training
            own_obs = agent_v

            # Look up the other agent oberservations to add to the training only
            # Look up the other agent actions to add to the training only
            other_obs, other_act = self.get_other_info(agent_k)

            if other_obs and other_act:
                new_observation_space[agent_k] = gymnasium.spaces.Dict(
                    OrderedDict({"own_obs": own_obs, "other_obs": other_obs, "other_act": other_act})
                )
            elif other_obs and not other_act:
                new_observation_space[agent_k] = gymnasium.spaces.Dict(OrderedDict({"own_obs": own_obs, "other_obs": other_obs}))
            elif not other_obs and other_act:
                new_observation_space[agent_k] = gymnasium.spaces.Dict(OrderedDict({"own_obs": own_obs, "other_act": other_act}))
            else:
                new_observation_space[agent_k] = agent_v  # type: ignore

        self._old_observation_space = self.observation_space
        self.observation_space = gymnasium.spaces.Dict(new_observation_space)  # type: ignore

        if self.config.observation_fn is None:
            self.config.observation_fn = CentralCriticObserver(
                self._agent_dict,
                self._trainable_agent_dict,
                self._non_trainable_agent_dict,
                self.config.process_nontrainables,
            )

    @staticmethod
    def get_validator() -> type[CorlCentralizedCriticEnvWrapperValidator]:
        """Get the validator for this class."""
        return CorlCentralizedCriticEnvWrapperValidator

    def _remove_trainable_agent_ids(self, agent_list: typing.Iterable[str]) -> list[str]:
        """
        Remove any agent IDs corresponding to trainable agents.
        """

        return [a for a in agent_list if a not in self._agent_ids_trainable]

    def _create_non_training_observations(self, alive_agents: typing.Iterable[str], observations: ObsBuffer) -> OrderedDict:
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

        alive_agents = self._remove_trainable_agent_ids(alive_agents)

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

        # TODO PUT THIS BACK?!?
        # if self.config.deep_sanity_check and self._episode_length % self.config.sanity_check_obs == 0:
        #     try:
        #         self._sanity_check(self._raw_observation_space, unit_extracted_obs)
        #     except ValueError as err:
        #         if self.config.save_state_pickle:
        #             self._save_state_pickle(err)

        return OrderedDict(
            self._agent_aggregator(
                agent_list=alive_agents,
                agent_function=lambda agent, agent_id, all_agent_obs: agent.create_training_observations(all_agent_obs[agent_id]),
                all_agent_obs=unit_extracted_obs,
            )
        )

    def reset(self, *, seed=None, options=None) -> tuple[OrderedDict, OrderedDict]:
        # Get the trainable items from the base environment
        training_obs, trainable_info = super().reset(seed=seed, options=options)
        # Generate observations for the other non trainable parts if enabled
        non_training_obs = None
        if self.config.process_nontrainables:
            # Populate the non trainable observations
            agent_list = list(self._agent_dict.keys())
            non_training_obs = self._create_non_training_observations(agent_list, self._obs_buffer)
        new_training_obs = self.config.observation_fn(
            training_obs, [self.action_space.sample()], self.observation_space, non_training_obs
        )  # type: ignore
        return new_training_obs, trainable_info

    def step(self, action_dict: dict):
        """Returns observations from ready agents."""
        # Get the trainable items from the base environment
        trainable_observations, trainable_rewards, trainable_dones, dones, trainable_info = super().step(action_dict)
        # Generate observations for the other non trainable parts if enabled
        non_training_obs = None
        if self.config.process_nontrainables:
            # Populate the non trainable observations
            agent_list = list(self._agent_dict.keys())
            non_training_obs = self._create_non_training_observations(agent_list, self._obs_buffer)
        return (
            self.config.observation_fn(trainable_observations, self._actions, self.observation_space, non_training_obs),  # type: ignore
            trainable_rewards,
            trainable_dones,
            dones,
            trainable_info,
        )

    def get_other_info(self, current_agent_str: str) -> tuple[gymnasium.spaces.Dict, gymnasium.spaces.Dict]:
        """Gets the observation and action space for other agents

        Args:
            current_agent_str (str): current agent which we are adding obs to

        Returns:
            tuple[gymnasium.spaces.Dict, gymnasium.spaces.Dict]: tuple containing both observation and action spaces for other agents
        """
        other_obs = OrderedDict()
        other_act = OrderedDict()
        # process all know agents to ensure we get privalaged information for the centralized critic
        for key in self._agent_dict:
            # ignore the key that is currently being augmented
            if key != current_agent_str:
                # Determine agent type
                if key in self._trainable_agent_dict:
                    other_obs[key] = self.observation_space[key]
                    other_act[key] = self.action_space[key]
                elif key in self._non_trainable_agent_dict:
                    if self.config.process_nontrainables:
                        other_obs[key] = self.full_observation_space[key]
                        other_act[key] = self._agent_dict[key].normalized_action_space
                    else:
                        warnings.warn(f"Augmenting info for {current_agent_str}: Skipping nontrainable agent for {key}!!!")
                else:
                    warnings.warn(f"Augmenting info for {current_agent_str}: Unable to add info for {key}!!!")

        return gymnasium.spaces.Dict(other_obs), gymnasium.spaces.Dict(other_act)
