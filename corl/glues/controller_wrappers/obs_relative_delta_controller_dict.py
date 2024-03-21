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

import gymnasium
import numpy as np
from pydantic import field_validator

from corl.glues.base_dict_wrapper import BaseDictWrapperGlue, BaseDictWrapperGlueValidator
from corl.glues.common.controller_glue import ControllerGlue
from corl.glues.common.observe_sensor import ObserveSensor
from corl.glues.controller_wrappers.obs_relative_delta_controller import RelativeObsDeltaAction
from corl.libraries.env_space_util import EnvSpaceUtil


class RelativeObsDeltaActionDictValidator(BaseDictWrapperGlueValidator):
    """
    step_size:      A dict that contains a floating point scalar for each action in the action space,
                    by which the corresponding delta action is scaled prior to converting the action
                    to the wrapped space.
                    e.g. A throttle DeltaAction.apply_action([0.2]) with step_size=[.05] would move the
                    absolute throttle position to 0.01 higher than it was at the end of the last step.
    """

    step_size: float = 1.0
    obs_index: int | None = 0
    is_wrap: bool = False
    initial_value: float | None = None

    @field_validator("step_size")
    @classmethod
    def check_step_scale(cls, v):
        """
        verifies range of step scale values
        """
        if v >= 1.0 or v < 0:
            raise ValueError("RelativeObsDeltaActionValidator got step size of more that 1.0 or less than 0")
        return v


class RelativeObsDeltaActionDict(BaseDictWrapperGlue, RelativeObsDeltaAction):  # type: ignore[misc]
    """
    RelativeObsDeltaActionDict is a glue class that wraps another glue class.
    It treats the actions passed to it as a delta from a linked observation
    E.G. if the wrapped action space has has roll as one of the controls, then a delta action of
    0.2 would move the absolute roll position 0.2 higher than it is as measured by the linked roll sensor.
    """

    def __init__(self, **kwargs) -> None:
        self.config: RelativeObsDeltaActionDictValidator  # type: ignore[assignment]
        BaseDictWrapperGlue.__init__(**kwargs)
        self._logger = logging.getLogger(RelativeObsDeltaActionDict.__name__)

        controller_keys = self.glues().keys()
        if "controller" not in controller_keys:
            raise KeyError("Missing key: controller")
        if "sensor" not in controller_keys:
            raise KeyError("Missing key: sensor")

        self.controller: ControllerGlue = typing.cast(ControllerGlue, self.glues()["controller"])
        if not isinstance(self.controller, ControllerGlue):
            raise RuntimeError(
                "Error: RelativeObsDeltaActionDict expects the glue wrapped on the 'controller' key to be a ControllerGlue, "
                f"got {self.controller}"
            )
        self.relative_obs_glue: ObserveSensor = typing.cast(ObserveSensor, self.glues()["sensor"])
        if not isinstance(self.relative_obs_glue, ObserveSensor):
            raise RuntimeError(
                "Error: RelativeObsDeltaActionDict expects the glue wrapped on the 'sensor' key to be a ObserveSensor, "
                f"got {self.relative_obs_glue}"
            )

        # verify that the config setup is not going to get the user into a situation where they are
        # only accessing one part of the obs but applying that obs as the base position for multiple actions

        tmp = self.controller.action_space
        if not isinstance(tmp, gymnasium.spaces.Dict):
            raise RuntimeError("obs relative delta controller only knows how to operate on dictionary action spaces currently")
        if self.config.obs_index and len(next(iter(tmp.spaces.values())).low) != 1:  # type: ignore
            raise RuntimeError(
                f"ERROR: your glue {self.get_unique_name()} has an action space length of more than 1, "
                "but you specified though obs_index to access only 1 component of the obs "
                "from the wrapped observe Sensor, to fix this error in your config for this glue define 'obs_index': null"
            )

        self.step_size = EnvSpaceUtil.convert_config_param_to_space(action_space=tmp, parameter=self.config.step_size)

        self._is_wrap = self.config.is_wrap

        self.saved_action_deltas = OrderedDict()
        assert isinstance(self.action_space, gymnasium.spaces.Dict)
        for space_name, space in self.action_space.items():
            if self.config.initial_value is not None:
                self.saved_action_deltas[space_name] = np.asarray([self.config.initial_value], dtype=np.float32)
            else:
                assert isinstance(space, gymnasium.spaces.Box)
                self.saved_action_deltas[space_name] = space.low

    @staticmethod
    def get_validator() -> type[RelativeObsDeltaActionDictValidator]:  # type: ignore[override]
        return RelativeObsDeltaActionDictValidator
