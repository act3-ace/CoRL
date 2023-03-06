"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Reward that uses episode state to accumulate reward"""
import abc
import enum
import typing
from collections import OrderedDict
from functools import partial

import numpy as np
from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, validator

from corl.dones.done_func_base import DoneStatusCodes
from corl.libraries.environment_dict import RewardDict
from corl.libraries.state_dict import StateDict
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator


class NegativeExponentialScaling(BaseModel):
    """Validation entry for negative exponential scaling."""
    scale: NonNegativeFloat
    eps: PositiveFloat


class EpisodeDoneRewardValidator(RewardFuncBaseValidator):
    """
    consolidate: if this done condition should attempt to reduce down to WLD in the case of
        multiple done conditions being set, with Win taking higher precedence
    skip_win_lose_sanity_check: notes if the done condition should skip the sanity check checking for
        both WIN and LOSS in an episodes done results, in case it was intentional
    """
    consolidate: bool = False
    skip_win_lose_sanity_check: bool = True


class EpisodeDoneReward(RewardFuncBase):
    """Base class for rewards that give rewards based on done information."""

    def __init__(self, **kwargs) -> None:
        self.config: EpisodeDoneRewardValidator
        super().__init__(**kwargs)
        self._counter = 0
        self._already_recorded: typing.Set[str] = set()
        self._status_codes: typing.Dict[DoneStatusCodes, typing.List[str]] = {x: [] for x in DoneStatusCodes}

    @property
    def get_validator(self) -> typing.Type[EpisodeDoneRewardValidator]:
        return EpisodeDoneRewardValidator

    @staticmethod
    def exp_scaling(x, *, scale: float, eps: float) -> float:
        """Scale as a negative exponential."""
        return scale * np.exp(-np.abs(x / eps))

    @staticmethod
    def constant_scaling(_, *, scale: float) -> float:
        """Scale by a constant."""
        return scale

    @staticmethod
    def zero_scaling(_) -> float:
        """Scale as zero."""
        return 0

    @abc.abstractmethod
    def get_scaling_method(self, done_name, done_code) -> typing.Callable[[int], float]:
        """Get the scaling method for a particular done name and code."""
        raise NotImplementedError()

    def __call__(
        self,
        observation: OrderedDict,
        action,
        next_observation: OrderedDict,
        state: StateDict,
        next_state: StateDict,
        observation_space,
        observation_units
    ) -> RewardDict:

        reward = RewardDict()
        reward[self.config.agent_name] = 0

        done_state = next_state.agent_episode_state.get(self.config.agent_name, {})
        for done_name, done_code in done_state.items():
            if done_name in self._already_recorded:
                continue
            self._already_recorded.add(done_name)
            self._status_codes[done_code].append(done_name)

        if not self.config.skip_win_lose_sanity_check:
            if len(self._status_codes[DoneStatusCodes.WIN]) > 0 and len(self._status_codes[DoneStatusCodes.LOSE]) > 0:
                raise RuntimeError(
                    "EpisodeDoneReward found both WIN and LOSS set during this episode, "
                    "if this is intended set skip_sanity_check=True"
                )

        # this will loop starting from win and go down
        consolidate_break = False
        for done_status in DoneStatusCodes:
            for done_name in self._status_codes[done_status]:
                reward[self.config.agent_name] += self.get_scaling_method(done_name, done_status)(self._counter)
                if self.config.consolidate:
                    consolidate_break = True
                    break
            if consolidate_break:
                break

        self._counter += 1 / next_state.sim_update_rate

        return reward


class EpisodeDoneStateRewardValidator(EpisodeDoneRewardValidator):
    """Validation for EpisodeDoneStateReward."""
    win: typing.Union[NegativeExponentialScaling, float] = 0
    partial_win: typing.Union[NegativeExponentialScaling, float] = 0
    draw: typing.Union[NegativeExponentialScaling, float] = 0
    partial_loss: typing.Union[NegativeExponentialScaling, float] = 0
    lose: typing.Union[NegativeExponentialScaling, float] = 0


class EpisodeDoneStateReward(EpisodeDoneReward):
    """Reward that responds to done condition state, once per done condition triggered."""

    def __init__(self, **kwargs) -> None:
        self.config: EpisodeDoneStateRewardValidator
        super().__init__(**kwargs)

        self._status_code_func = {}
        for code in DoneStatusCodes:
            code_name = code.name.lower()
            if not hasattr(self.config, code_name):
                raise RuntimeError(f'Unknown done status code: {code_name}')
            code_data = getattr(self.config, code_name)
            if isinstance(code_data, NegativeExponentialScaling):
                self._status_code_func[code_name] = partial(self.exp_scaling, scale=code_data.scale, eps=code_data.eps)
            else:
                self._status_code_func[code_name] = partial(self.constant_scaling, scale=code_data)

    @property
    def get_validator(self) -> typing.Type[EpisodeDoneStateRewardValidator]:
        return EpisodeDoneStateRewardValidator

    def get_scaling_method(self, done_name, done_code) -> typing.Callable[[int], float]:
        return self._status_code_func[done_code.name.lower()]


class _MissingMethod(enum.Enum):
    zero = 'zero'
    raise_exception = 'raise'


class EpisodeDoneNameRewardValidator(EpisodeDoneRewardValidator):
    """Validation for EpisodeDoneNameReward."""
    rewarded_dones: typing.Dict[str, typing.Union[NegativeExponentialScaling, float]] = {}
    missing_method: _MissingMethod = _MissingMethod.zero

    @validator('rewarded_dones', always=True)
    def not_empty(cls, v):
        """Ensure that some done condition is specified."""
        if len(v) == 0:
            raise ValueError('Rewarded dones cannot be empty')
        return v


class EpisodeDoneNameReward(EpisodeDoneReward):
    """Reward that responds once to individual dones."""

    def __init__(self, **kwargs) -> None:
        self.config: EpisodeDoneNameRewardValidator
        super().__init__(**kwargs)

        self._done_name_func = {}
        for name, args in self.config.rewarded_dones.items():
            if isinstance(args, NegativeExponentialScaling):
                self._done_name_func[name] = partial(self.exp_scaling, scale=args.scale, eps=args.eps)
            else:
                self._done_name_func[name] = partial(self.constant_scaling, scale=args)

    @property
    def get_validator(self) -> typing.Type[EpisodeDoneNameRewardValidator]:
        return EpisodeDoneNameRewardValidator

    def get_scaling_method(self, done_name, done_code) -> typing.Callable[[int], float]:
        if self.config.missing_method == _MissingMethod.zero:
            return self._done_name_func.get(done_name, self.zero_scaling)

        return self._done_name_func[done_name]
