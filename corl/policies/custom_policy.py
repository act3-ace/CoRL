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
import typing
from abc import abstractmethod

import flatten_dict
import gym
import numpy as np
from pydantic import validator
from ray.rllib.evaluation import Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType

from corl.libraries.env_space_util import EnvSpaceUtil
from corl.policies.base_policy import BasePolicyValidator
from corl.rewards.base_measurement_operation import ObservationExtractorValidator


class CustomPolicyValidator(BasePolicyValidator):
    """Base validator for the CustomPolicy"""

    act_space: gym.Space
    obs_space: gym.Space

    time_extractor: ObservationExtractorValidator

    # rllib assumes that actions have been normalized and calls 'unsquash_action' prior to sending it to the environment:
    # https://github.com/ray-project/ray/blob/c78bd809ce4b2ec0e48c77aa461684ee1e6f259b/rllib/evaluation/sampler.py#L1241
    normalize_controls: bool = True

    controllers: typing.List[typing.Tuple]

    class Config:
        """pydantic configuration options"""
        arbitrary_types_allowed = True

    @validator('act_space')
    def validate_act_space(cls, v):  # pylint: disable=no-self-argument, no-self-use
        """validate that it's an instance of an gym.Space"""
        assert isinstance(v, gym.Space)
        # TODO Issue warning if the action space is normalized
        return v

    @validator('time_extractor', always=True)
    def validate_extractor(cls, v, values):  # pylint: disable=no-self-argument, no-self-use
        """Ensures the time_extractor can actually extract data from the space and that it's not normalized"""
        try:
            time_space = v.construct_extractors().space(values['obs_space'].original_space)
            if isinstance(time_space, gym.spaces.Box):
                assert np.isinf(time_space.high[v.indices[0]]), "time_space must not be normalized"

        except Exception as e:
            raise RuntimeError(f"Failed to extract time using {v} from {values['obs_space'].original_space}") from e

        return v

    @validator('controllers', pre=True, always=True)
    def validate_controllers(cls, v, values):  # pylint: disable=no-self-argument, no-self-use
        """validate that the controllers match the action_space"""
        tuple_list = []
        for iterable in v:
            tuple_list.append(tuple(iterable))

        assert len(tuple_list) == len(set(tuple_list)), 'controller definitions must be unique'

        sample_control = flatten_dict.flatten(values['act_space'].sample())

        for tuple_key in tuple_list:
            assert tuple_key in sample_control, f'controller {tuple_key} not found in action_space: {list(sample_control.keys())}'

        return tuple_list


class CustomPolicy(Policy):  # pylint: disable=abstract-method
    """Custom base policy.
    """

    def __init__(self, observation_space, action_space, config):
        self.validated_config: CustomPolicyValidator = self.get_validator(act_space=action_space, obs_space=observation_space, **config)

        Policy.__init__(self, observation_space, action_space, config)

        self.time_extractor = self.validated_config.time_extractor.construct_extractors()
        self._reset()

    @property
    def get_validator(self) -> typing.Type[BasePolicyValidator]:
        """
        Get the validator for this experiment class,
        the kwargs sent to the experiment class will
        be validated using this object and add a self.config
        attr to the experiment class
        """
        return CustomPolicyValidator

    def _reset(self):
        """This must be overriden in order to reset the state between runs
        """
        ...

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        return

    def compute_actions_from_input_dict(
        self,
        input_dict: typing.Union[SampleBatch, typing.Dict[str, TensorStructType]],
        explore: bool = None,
        timestep: typing.Optional[int] = None,
        episodes: typing.Optional[typing.List[Episode]] = None,
        **kwargs
    ) -> typing.Tuple[TensorType, typing.List[TensorType], typing.Dict[str, TensorType]]:
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
        agent_index = input_dict[SampleBatch.AGENT_INDEX][0]
        episode_id = input_dict[SampleBatch.EPS_ID][0]
        episode = [eps for eps in episodes if eps.episode_id == episode_id][0]  # type: ignore
        agent_id: str = [
            aid for aid in episode._agent_to_index if episode._agent_to_index[aid] == agent_index  # pylint: disable=protected-access
        ][0]

        obs_batch = input_dict[SampleBatch.OBS]
        info = episode.last_info_for(agent_id)
        if info is None:
            info = {}

        if 'platform_obs' in info:
            sim_time = self.time_extractor.value(info['platform_obs'][agent_id], full_extraction=True)
        else:
            self._reset()
            sim_time = -1

        state_batches = [s for k, s in input_dict.items() if k[:9] == "state_in_"]
        return self.compute_actions(
            obs_batch,
            state_batches,
            prev_action_batch=input_dict.get(SampleBatch.PREV_ACTIONS),
            prev_reward_batch=input_dict.get(SampleBatch.PREV_REWARDS),
            info_batch=input_dict.get(SampleBatch.INFOS),  # type: ignore
            explore=explore,
            timestep=timestep,
            episodes=episodes,
            sim_time=sim_time,
            agent_id=agent_id,
            info=info,
            episode=episode,
            **kwargs,
        )

    def compute_actions(
        self,
        obs_batch: typing.Union[typing.List[TensorStructType], TensorStructType],
        state_batches: typing.Optional[typing.List[TensorType]] = None,
        prev_action_batch: typing.Union[typing.List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: typing.Union[typing.List[TensorStructType], TensorStructType] = None,
        info_batch: typing.Optional[typing.Dict[str, list]] = None,
        episodes: typing.Optional[typing.List[Episode]] = None,
        explore: typing.Optional[bool] = None,
        timestep: typing.Optional[int] = None,
        **kwargs
    ) -> typing.Tuple[TensorType, typing.List[TensorType], typing.Dict[str, TensorType]]:
        actions, state_outs, info = self.custom_compute_actions(
            obs_batch,
            state_batches=state_batches,
            prev_action_batch=prev_action_batch,
            prev_reward_batch=prev_reward_batch,
            info_batch=info_batch,
            episodes=episodes,
            explore=explore,
            timestep=timestep,
            **kwargs
        )
        if self.validated_config.normalize_controls:
            for i, action in enumerate(actions):
                actions[i] = EnvSpaceUtil.scale_sample_from_space(self.validated_config.act_space, action)
        return actions, state_outs, info

    @abstractmethod
    def custom_compute_actions(
        self,
        obs_batch: typing.Union[typing.List[TensorStructType], TensorStructType],
        state_batches: typing.Optional[typing.List[TensorType]] = None,
        prev_action_batch: typing.Union[typing.List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: typing.Union[typing.List[TensorStructType], TensorStructType] = None,
        info_batch: typing.Optional[typing.Dict[str, list]] = None,
        episodes: typing.Optional[typing.List[Episode]] = None,
        explore: typing.Optional[bool] = None,
        timestep: typing.Optional[int] = None,
        sim_time: typing.Optional[float] = None,
        agent_id: typing.Optional[str] = None,
        info: typing.Optional[dict] = None,
        episode: typing.Optional[Episode] = None,
        **kwargs
    ) -> typing.Tuple[TensorType, typing.List[TensorType], typing.Dict[str, TensorType]]:
        """Computes actions for the current policy.

        Args:
            obs_batch: Batch of observations.
            state_batches: List of RNN state input batches, if any.
            prev_action_batch: Batch of previous action values.
            prev_reward_batch: Batch of previous rewards.
            info_batch: Batch of info objects.
            episodes: List of Episode objects, one for each obs in
                obs_batch. This provides access to all of the internal
                episode state, which may be useful for model-based or
                multi-agent algorithms.
            explore: Whether to pick an exploitation or exploration action.
                Set to None (default) for using the value of
                `self.config["explore"]`.
            timestep: The current (sampling) time step.

        Keyword Args:
            kwargs: Forward compatibility placeholder

        Returns:
            actions (TensorType): Batch of output actions, with shape like
                [BATCH_SIZE, ACTION_SHAPE].
            state_outs (List[TensorType]): List of RNN state output
                batches, if any, each with shape [BATCH_SIZE, STATE_SIZE].
            info (List[dict]): Dictionary of extra feature batches, if any,
                with shape like
                {"f1": [BATCH_SIZE, ...], "f2": [BATCH_SIZE, ...]}.
        """
        raise NotImplementedError
