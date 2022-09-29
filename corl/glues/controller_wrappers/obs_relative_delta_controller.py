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
import numpy as np
from pydantic import validator

from corl.glues.base_multi_wrapper import BaseMultiWrapperGlue, BaseMultiWrapperGlueValidator
from corl.glues.common.controller_glue import ControllerGlue
from corl.glues.common.observe_sensor import ObserveSensor
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.property import BoxProp
from corl.libraries.units import Convert


class RelativeObsDeltaActionValidator(BaseMultiWrapperGlueValidator):
    """
    step_size:      A dict that contains a floating point scalar for each action in the action space,
                    by which the corresponding delta action is scaled prior to converting the action
                    to the wrapped space.
                    e.g. A throttle DeltaAction.apply_action([0.2]) with step_size=[.05] would move the
                    absolute throttle position to 0.01 higher than it was at the end of the last step.
    """
    step_size: float = 1.0
    obs_index: typing.Optional[int] = 0
    is_wrap: bool = False
    initial_value: typing.Optional[float] = None

    @validator("step_size")
    @classmethod
    def check_step_scale(cls, v):
        """
        verifies range of step scale values
        """
        if v >= 1.0 or v < 0:
            raise ValueError("RelativeObsDeltaActionValidator got step size of more that 1.0 or less than 0")
        return v


class RelativeObsDeltaAction(BaseMultiWrapperGlue):
    """
    RelativeObsDeltaAction is a glue class that wraps another glue class.
    It treats the actions passed to it as a delta from a linked observation
    E.G. if the wrapped action space has has roll as one of the controls, then a delta action of
    0.2 would move the absolute roll position 0.2 higher than it is as measured by the linked roll sensor.
    """

    def __init__(self, **kwargs) -> None:
        self.config: RelativeObsDeltaActionValidator
        super().__init__(**kwargs)

        self._logger = logging.getLogger(RelativeObsDeltaAction.__name__)

        if len(self.glues()) != 2:
            raise RuntimeError(f"Error: RelativeObsDeltaAction expected 2 wrapped glues, got {len(self.glues())}")
        self.controller: ControllerGlue = typing.cast(ControllerGlue, self.glues()[0])
        if not isinstance(self.controller, ControllerGlue):
            raise RuntimeError(
                f"Error: RelativeObsDeltaAction expects the first wrapped glue to be a ControllerGlue, got {self.controller}"
            )
        self.relative_obs_glue: ObserveSensor = typing.cast(ObserveSensor, self.glues()[1])
        if not isinstance(self.relative_obs_glue, ObserveSensor):
            raise RuntimeError(
                f"Error: RelativeObsDeltaAction expects the second wrapped glue to be a ObserveSensor, got {self.relative_obs_glue}"
            )

        # verify that the config setup is not going to get the user into a situation where they are
        # only accessing one part of the obs but applying that obs as the base position for multiple actions
        if self.config.obs_index and len(list(self.controller.action_space().values())[0].low) != 1:
            raise RuntimeError(
                f"ERROR: your glue {self.get_unique_name()} has an action space length of more than 1, "
                "but you specified though obs_index to access only 1 component of the obs "
                "from the wrapped observe Sensor, to fix this error in your config for this glue define 'obs_index': null"
            )

        self.step_size = EnvSpaceUtil.convert_config_param_to_space(
            action_space=self.controller.action_space(), parameter=self.config.step_size
        )

        self._is_wrap = self.config.is_wrap

        self.saved_action_deltas = OrderedDict()
        for space_name, space in self.action_space().items():
            if self.config.initial_value is not None:
                self.saved_action_deltas[space_name] = np.asarray([self.config.initial_value], dtype=np.float32)
            else:
                self.saved_action_deltas[space_name] = space.low

    @property
    def get_validator(self) -> typing.Type[RelativeObsDeltaActionValidator]:
        return RelativeObsDeltaActionValidator

    @lru_cache()
    def get_unique_name(self) -> str:
        """Class method that retreives the unique name for the glue instance
        """
        wrapped_glue_name = self.controller.get_unique_name()
        if wrapped_glue_name is None:
            return None
        return wrapped_glue_name + "RelativeDelta"

    def get_observation(self) -> typing.Union[np.ndarray, typing.Tuple, typing.Dict]:
        return {
            "absolute": self.controller.get_observation(),
            "delta": self.saved_action_deltas,
        }

    @lru_cache()
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Dict({"absolute": self.controller.observation_space(), "delta": self.action_space()})

    @lru_cache()
    def action_space(self) -> gym.spaces.Space:
        """
        Build the action space for the controller, etc.
        """

        # get the action space from the parent
        original_action_space = self.controller.action_space()

        # log the original action space
        self._logger.debug(f"action_space: {original_action_space}")

        # zero mean the space so we can scale it easier
        zero_mean_space = EnvSpaceUtil.zero_mean_space(original_action_space)

        # scale the size of the unbiased space
        for space_name, space in zero_mean_space.items():
            zero_mean_space[space_name] = EnvSpaceUtil.scale_space(space, scale=self.step_size[space_name])

        return zero_mean_space

    # TODO: assumes self.controller._control_properties has unit attribute
    def apply_action(self, action, observation) -> None:
        """
        Apply the action for the controller, etc.
        """

        self._logger.debug(f"apply_action: {action}")

        current_observation = self.relative_obs_glue.get_observation()["direct_observation"]
        # all units in an array must be the same, so this assumption is ok
        obs_units = self.relative_obs_glue.observation_units()["direct_observation"][0]
        if self.config.obs_index:
            current_observation = current_observation[self.config.obs_index]
        assert isinstance(self.controller._control_properties, BoxProp), "Unexpected control_properties type"  # pylint: disable=W0212
        out_unit = self.controller._control_properties.unit[0]  # pylint: disable=W0212
        assert isinstance(out_unit, str)
        unit_converted_obs = Convert(current_observation, obs_units, out_unit)

        new_base_obs = OrderedDict()
        for control in action.keys():
            new_base_obs[control] = unit_converted_obs

        self.saved_action_deltas = action

        absolute_action = EnvSpaceUtil.add_space_samples(
            space_template=self.action_space(),
            space_sample1=action,
            space_sample2=new_base_obs,
        )
        absolute_action = EnvSpaceUtil.clip_space_sample_to_space(absolute_action, self.controller.action_space(), self._is_wrap)

        try:
            self.controller.apply_action(absolute_action, observation)
        except Exception as exc:
            # Purpose - add additional debugging information and re-raise the exception
            raise ValueError(
                f'\n'
                f'action={action}\n'
                f'current_observation={current_observation}\n'
                f'obs_unit={obs_units}\n'
                f'out_unit={out_unit}\n'
                f'action_space={self.action_space()}\n'
                f'controller_action_space={self.controller.action_space()}\n'
                f'is_wrap={self._is_wrap}\n'
            ) from exc

    def get_info_dict(self):
        """
        Get the user specified metadata/metrics/etc.
        """
        return {}
