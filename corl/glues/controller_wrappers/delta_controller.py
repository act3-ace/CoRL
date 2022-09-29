"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

AvailablePlatforms
"""
import logging
import typing
from collections import OrderedDict
from functools import lru_cache

import gym
from pydantic import validator

from corl.glues.base_glue import BaseAgentControllerGlue
from corl.glues.base_wrapper import BaseWrapperGlue, BaseWrapperGlueValidator
from corl.libraries.env_space_util import EnvSpaceUtil


class DeltaActionValidator(BaseWrapperGlueValidator):
    """
    step_size:      A dict that contains a floating point scalar for each action in the action space,
                    by which the corresponding delta action is scaled prior to converting the action
                    to the wrapped space.
                    e.g. A throttle DeltaAction.apply_action([0.2]) with step_size=[.05] would move the
                    absolute throttle position to 0.01 higher than it was at the end of the last step.
    """
    step_size: float = 1.0
    is_wrap: bool

    @validator("step_size")
    @classmethod
    def check_step_scale(cls, v):
        """
        verifies range of step scale values
        """
        if v >= 1.0 or v < 0:
            raise ValueError("DeltaActionValidator got step size of more that 1.0 or less than 0")
        return v


class DeltaAction(BaseWrapperGlue):
    """
    DeltaAction is a glue class that wraps another glue class.
    It treats the actions passed to it as a delta from the last action command rather
    E.G. if the wrapped action space has has throttle as one of the controls, then a delta action of
    0.2 would move the absolute throttle position 0.2 higher than it was at the end of the last step.
    """

    def __init__(self, **kwargs) -> None:
        self.config: DeltaActionValidator
        super().__init__(**kwargs)

        self._logger = logging.getLogger(DeltaAction.__name__)

        self.step_size = EnvSpaceUtil.convert_config_param_to_space(
            action_space=self.glue().action_space(), parameter=self.config.step_size
        )

        self._is_wrap = self.config.is_wrap

        self.saved_action_deltas: EnvSpaceUtil.sample_type = OrderedDict()
        for space_name, space in self.action_space().items():
            self.saved_action_deltas[space_name] = space.low

        inner_glue: BaseAgentControllerGlue = typing.cast(BaseAgentControllerGlue, self.glue())
        if not isinstance(inner_glue, BaseAgentControllerGlue):
            raise TypeError(f"Inner glue is not a BaseAgentControllerGlue, but rather {type(inner_glue).__name__}")

    @property
    def get_validator(self) -> typing.Type[DeltaActionValidator]:
        """Return validator"""
        return DeltaActionValidator

    @lru_cache()
    def get_unique_name(self) -> str:
        """Class method that retreives the unique name for the glue instance
        """
        wrapped_glue_name = self.glue().get_unique_name()
        if wrapped_glue_name is None:
            return None
        return wrapped_glue_name + "Delta"

    def get_observation(self) -> EnvSpaceUtil.sample_type:
        """Get observation"""
        return {"absolute": self.glue().get_observation(), "delta": self.saved_action_deltas}

    @lru_cache()
    def observation_space(self) -> gym.spaces.Space:
        """Observation space"""
        return gym.spaces.Dict({"absolute": self.glue().observation_space(), "delta": self.action_space()})

    @lru_cache()
    def action_space(self) -> gym.spaces.Space:
        """
        Build the action space for the controller, etc.
        """

        # get the action space from the parent
        original_action_space = self.glue().action_space()

        # log the original action space
        self._logger.debug(f"action_space: {original_action_space}")

        # zero mean the space so we can scale it easier
        zero_mean_space = EnvSpaceUtil.zero_mean_space(original_action_space)

        # scale the size of the unbiased space
        for space_name, space in zero_mean_space.items():
            zero_mean_space[space_name] = EnvSpaceUtil.scale_space(space, scale=self.step_size[space_name])

        return zero_mean_space

    def apply_action(self, action: EnvSpaceUtil.sample_type, observation: EnvSpaceUtil.sample_type) -> None:
        """
        Apply the action for the controller, etc.
        """

        self._logger.debug(f"apply_action: {action}")

        inner_glue: BaseAgentControllerGlue = typing.cast(BaseAgentControllerGlue, self.glue())
        if not isinstance(inner_glue, BaseAgentControllerGlue):
            raise TypeError(f"Inner glue is not a BaseAgentControllerGlue, but rather {type(inner_glue).__name__}")

        last_absolute_action = inner_glue.get_applied_control()

        absolute_action = EnvSpaceUtil.add_space_samples(
            space_template=self.action_space(),
            space_sample1=action,
            space_sample2=last_absolute_action,
        )
        absolute_action = EnvSpaceUtil.clip_space_sample_to_space(absolute_action, inner_glue.action_space(), self._is_wrap)

        self.saved_action_deltas = action

        inner_glue.apply_action(absolute_action, observation)

    def get_info_dict(self):
        """
        Get the user specified metadata/metrics/etc.
        """
        return {}
