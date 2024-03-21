"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Custom Policy
"""
from abc import abstractmethod
from collections import defaultdict

import flatten_dict
import gymnasium
import numpy as np
from pydantic import ConfigDict, model_validator
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.typing import TensorStructType, TensorType

from corl.policies.base_policy import BasePolicyValidator


class CustomPolicyValidator(BasePolicyValidator):
    """Base validator for the CustomPolicy"""

    agent_id: str | None = None

    act_space: gymnasium.Space
    obs_space: gymnasium.Space
    controllers: list[tuple[str, ...]]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_model(self):
        assert len(self.controllers) == len(set(self.controllers)), "controller definitions must be unique"
        sample_control = flatten_dict.flatten(self.act_space.sample())
        for tuple_key in self.controllers:
            assert tuple_key in sample_control, f"controller {tuple_key} not found in action_space: {list(sample_control.keys())}"
        return self


class CustomPolicy(Policy):
    """Custom base policy."""

    def __init__(self, observation_space, action_space, config) -> None:
        self.validated_config: CustomPolicyValidator = self.get_validator()(act_space=action_space, obs_space=observation_space, **config)

        Policy.__init__(self, observation_space, action_space, config)

        self._agent_id = self.validated_config.agent_id
        self._reset_time = 0

        self._reset()

    @staticmethod
    def get_validator() -> type[CustomPolicyValidator]:
        """
        Get the validator for this experiment class,
        the kwargs sent to the experiment class will
        be validated using this object and add a self.config
        attr to the experiment class
        """
        return CustomPolicyValidator

    def _reset(self):
        """This must be overridden in order to reset the state between runs"""

    def learn_on_batch(self, samples):  # noqa: PLR6301
        return {}

    def get_weights(self):  # noqa: PLR6301
        return {}

    def set_weights(self, weights):  # noqa: PLR6301
        return

    def compute_actions_from_input_dict(
        self,
        input_dict: SampleBatch | dict[str, TensorStructType],
        explore: bool | None = None,
        timestep: int | None = None,
        episodes: list[EpisodeV2] | None = None,
        **kwargs,
    ) -> tuple[TensorType, list[TensorType], dict[str, TensorType]]:
        """Computes actions from collected samples (across multiple-agents).

        Takes an input dict (usually a SampleBatch) as its main data input.
        This allows for using this method in case a more complex input pattern
        (view requirements) is needed, for example when the Model requires the
        last n observations, the last m actions/rewards, or a combination
        of any of these.

        Args:
            input_dict: A SampleBatch or input dict containing the Tensors
                to compute actions. `input_dict` already abides to the
                Policy's as well as the Model's view requirements and can
                thus be passed to the Model as-is.
            explore: Whether to pick an exploitation or exploration
                action (default: None -> use self.config["explore"]).
            timestep: The current (sampling) time step.
            episodes: This provides access to all of the internal episodes'
                state, which may be useful for model-based or multi-agent
                algorithms.

        Keyword Args:
            kwargs: Forward compatibility placeholder.

        Returns:
            actions: Batch of output actions, with shape like
                [BATCH_SIZE, ACTION_SHAPE].
            state_outs: List of RNN state output
                batches, if any, each with shape [BATCH_SIZE, STATE_SIZE].
            info: Dictionary of extra feature batches, if any, with shape like
                {"f1": [BATCH_SIZE, ...], "f2": [BATCH_SIZE, ...]}.
        """
        # Default implementation just passes obs, prev-a/r, and states on to
        # `self.compute_actions()`.
        agent_id_batch = []
        raw_obs_batch = []
        epp_info_batch = []
        sim_time_batch = []
        do_reset = False

        for i in range(input_dict.count):  # type: ignore
            episode = episodes[i]  # type: ignore

            # there is a really nasty bug in the agent id code and it comes out as -1 sometimes
            # so we save a valid one and just cache it so we don't have to worry about it
            agent_index = input_dict[SampleBatch.AGENT_INDEX][i]
            agents = episode.get_agents()
            self._agent_id = agents[agent_index] if 0 <= agent_index < len(agents) else self._agent_id
            if self._agent_id is None:
                raise RuntimeError(f"Unable to determine the agent_id for agent_index: {agent_index}")
            # end hack section

            epp_info = episode.last_info_for(self._agent_id)
            if epp_info:
                sim_time = epp_info["env"]["sim_time"]
            else:
                epp_info = {}
                sim_time = self._reset_time

            if episode.length == 0 or sim_time < self._reset_time:
                self._reset_time = sim_time

            if sim_time <= self._reset_time:
                do_reset = True

            # rllib preprocessor add original space attribute to the space
            raw_obs = gymnasium.spaces.utils.unflatten(self.observation_space.original_space, input_dict["obs"][i])  # type: ignore

            sim_time_batch.append(sim_time)
            agent_id_batch.append(self._agent_id)
            epp_info_batch.append(epp_info)
            raw_obs_batch.append(raw_obs)

        if do_reset:
            self._reset()

        return self.compute_actions(
            input_dict[SampleBatch.OBS],
            [s for k, s in input_dict.items() if k[:9] == "state_in_"],
            prev_action_batch=input_dict.get(SampleBatch.PREV_ACTIONS),
            prev_reward_batch=input_dict.get(SampleBatch.PREV_REWARDS),
            info_batch=input_dict.get(SampleBatch.INFOS),  # type: ignore
            explore=explore,
            timestep=timestep,
            episodes=episodes,
            sim_time_batch=sim_time_batch,
            agent_id_batch=agent_id_batch,
            epp_info_batch=epp_info_batch,
            raw_obs_batch=raw_obs_batch,
            **kwargs,
        )

    def compute_actions(
        self,
        obs_batch: list[TensorStructType] | TensorStructType,
        state_batches: list[TensorType] | None = None,
        prev_action_batch: list[TensorStructType] | TensorStructType = None,
        prev_reward_batch: list[TensorStructType] | TensorStructType = None,
        info_batch: dict[str, list] | None = None,
        episodes: list[EpisodeV2] | None = None,
        explore: bool | None = None,
        timestep: int | None = None,
        **kwargs,
    ) -> tuple[TensorType, list[TensorType], dict[str, TensorType]]:
        num_batches = len(obs_batch)

        sim_time_batch = kwargs.get("sim_time_batch", None)
        agent_id_batch = kwargs.get("agent_id_batch", None)
        epp_info_batch = kwargs.get("epp_info_batch", None)
        raw_obs_batch = kwargs.get("raw_obs_batch", [{} for _ in range(num_batches)])

        actions_batch = defaultdict(list)

        for i in range(num_batches):
            actions = self.custom_compute_actions(
                obs_batch[i],  # type: ignore
                platform_obs=raw_obs_batch[i],
                state=state_batches[i] if state_batches is not None and len(state_batches) > i else None,
                prev_action=prev_action_batch[i] if prev_action_batch is not None and len(prev_action_batch) > i else None,
                prev_reward=prev_reward_batch[i] if prev_reward_batch is not None and len(prev_reward_batch) > i else None,
                info=info_batch[i] if info_batch is not None and len(info_batch) > i else None,  # type: ignore
                explore=explore,
                timestep=timestep,
                sim_time=sim_time_batch[i] if sim_time_batch is not None and len(sim_time_batch) > i else None,
                agent_id=agent_id_batch[i] if agent_id_batch is not None and len(agent_id_batch) > i else None,
                epp_info=epp_info_batch[i] if epp_info_batch is not None and len(epp_info_batch) > i else None,
                episode=episodes[i] if episodes is not None and len(episodes) > i else None,
                **kwargs,
            )

            if self.config.get("normalize_actions"):
                actions = normalize_action(action=actions, action_space_struct=self.validated_config.act_space)

            for key, value in flatten_dict.flatten(actions).items():
                actions_batch[key].append(value)

        for key, value in actions_batch.items():
            actions_batch[key] = np.array(value)  # type: ignore

        # # NOTE: ray/rllib is inconsistent in applying these modifications to the returned actions (between training and the
        # # Algorithm.compute_single_action fn) so we do it explicitly here to ensure that the behavior is consistent between them.
        # actions = convert_to_numpy(actions)
        # if isinstance(actions, list):
        #     actions = np.array(actions)

        return flatten_dict.unflatten(actions_batch), [], {}

    @abstractmethod
    def custom_compute_actions(
        self,
        obs: list[TensorStructType] | TensorStructType,
        platform_obs: dict[str, dict],
        state: list[TensorType] | None = None,
        prev_action: list[TensorStructType] | TensorStructType = None,
        prev_reward: list[TensorStructType] | TensorStructType = None,
        info: dict[str, list] | None = None,
        explore: bool | None = None,
        timestep: int | None = None,
        sim_time: float | None = None,
        agent_id: str | None = None,
        epp_info: dict | None = None,
        episode: EpisodeV2 | None = None,
        **kwargs,
    ) -> TensorType:
        """Computes actions for the current policy.

        Args:
            obs: observations.
            state: List of RNN state input batches, if any.
            prev_action: previous action values.
            prev_reward: previous rewards.
            info: info objects.
            explore: Whether to pick an exploitation or exploration action.
                Set to None (default) for using the value of
                `self.config["explore"]`.
            timestep: The current (sampling) time step.

        Keyword Args:
            kwargs: Forward compatibility placeholder

        Returns:
            actions (TensorType): output actions
        """
        raise NotImplementedError
