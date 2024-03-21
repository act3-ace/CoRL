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
from functools import cached_property

import numpy as np
from gymnasium.spaces import Dict
from pydantic import field_validator

from corl.glues.base_glue import BaseAgentControllerGlue
from corl.glues.base_wrapper import BaseWrapperGlue, BaseWrapperGlueValidator
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.property import DictProp
from corl.libraries.units import corl_get_ureg


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

    @field_validator("step_size")
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

        wrapped_aspace = self.glue().action_space
        assert wrapped_aspace is not None
        self.step_size = EnvSpaceUtil.convert_config_param_to_space(action_space=wrapped_aspace, parameter=self.config.step_size)

        self._is_wrap = self.config.is_wrap

        self.saved_action_deltas: EnvSpaceUtil.sample_type = OrderedDict()
        assert isinstance(self.action_space, Dict)
        for space_name in self.action_space:
            self.saved_action_deltas[space_name] = corl_get_ureg().Quantity(
                np.array([0], dtype=self.action_prop.spaces[space_name].dtype), self.action_prop.spaces[space_name].get_units()
            )

        inner_glue: BaseAgentControllerGlue = typing.cast(BaseAgentControllerGlue, self.glue())
        if not isinstance(inner_glue, BaseAgentControllerGlue):
            raise TypeError(f"Inner glue is not a BaseAgentControllerGlue, but rather {type(inner_glue).__name__}")

        wrapped_glue_name = self.glue().get_unique_name()
        self._uname = None if wrapped_glue_name is None else f"{wrapped_glue_name}Delta"

    @staticmethod
    def get_validator() -> type[DeltaActionValidator]:
        """Return validator"""
        return DeltaActionValidator

    def get_unique_name(self) -> str:
        """Class method that retrieves the unique name for the glue instance"""
        return self._uname

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> EnvSpaceUtil.sample_type:
        """Get observation"""
        return {"absolute": self.glue().get_observation(other_obs, obs_space, obs_units), "delta": self.saved_action_deltas}

    @cached_property
    def observation_prop(self):
        return DictProp(spaces={"absolute": self.glue().observation_prop, "delta": self.action_prop})

    @cached_property
    def action_prop(self):
        # get the action space from the parent
        original_action_space = self.glue().action_prop
        assert original_action_space is not None
        # zero mean the space so we can scale it easier
        zero_mean_space = original_action_space.zero_mean()

        # scale the size of the unbiased space
        for space_name, space in zero_mean_space.spaces.items():
            zero_mean_space.spaces[space_name] = space.scale(scale=self.step_size[space_name])

        return zero_mean_space

    def apply_action(
        self, action: EnvSpaceUtil.sample_type, observation: EnvSpaceUtil.sample_type, action_space, obs_space, obs_units
    ) -> None:
        """
        Apply the action for the controllers
        """

        self._logger.debug(f"apply_action: {action}")

        inner_glue: BaseAgentControllerGlue = typing.cast(BaseAgentControllerGlue, self.glue())
        if not isinstance(inner_glue, BaseAgentControllerGlue):
            raise TypeError(f"Inner glue is not a BaseAgentControllerGlue, but rather {type(inner_glue).__name__}")

        last_absolute_action = inner_glue.get_applied_control()

        assert self.action_space is not None
        absolute_action = EnvSpaceUtil.add_space_samples(
            space_template=self.action_space,
            space_sample1=action,
            space_sample2=last_absolute_action,
        )
        assert inner_glue.action_space is not None
        absolute_action = EnvSpaceUtil.clip_space_sample_to_space(absolute_action, inner_glue.action_space, self._is_wrap)

        self.saved_action_deltas = action

        inner_glue.apply_action(absolute_action, observation, action_space, obs_space, obs_units)

    def get_info_dict(self):  # noqa: PLR6301
        """
        Get the user specified metadata/metrics/etc.
        """
        return {}
