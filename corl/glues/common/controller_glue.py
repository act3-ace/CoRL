# pylint: disable=no-member
"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Controller Glue
"""
import logging
import typing
from collections import OrderedDict
from functools import lru_cache

import gym

from corl.glues.base_glue import BaseAgentControllerGlue, BaseAgentPlatformGlueValidator
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.simulators.base_parts import BaseController
from corl.simulators.common_platform_utils import get_controller_by_name

code_logger = logging.getLogger("code_timing")


class ControllerGlueValidator(BaseAgentPlatformGlueValidator):
    """
    controller: Which controller to get from parent platform
    minimum: temporary value to override the minumum for this glue
    maximum: temporary value to override the minimum for this glue
    index: if you are using more than one of the same controller,
            use this to access which index to use
    """
    controller: str


class ControllerGlue(BaseAgentControllerGlue):
    """
    This simple glue class wraps a controller and creates an action space based on the controller exclusiveness
    This class has no observation space or observations
    """

    def __init__(self, **kwargs) -> None:
        self.config: ControllerGlueValidator
        super().__init__(**kwargs)

        self._controller = get_controller_by_name(self._platform, self.config.controller)
        self._controller_name: str = self.config.controller
        self._key = self._controller.control_properties.name
        self._control_properties = self._controller.control_properties

    @property
    def controller(self) -> BaseController:
        """Returns controller

        Returns
        -------
        BaseController
            The controller for this glue
        """
        return self._controller

    @property
    def get_validator(self) -> typing.Type[ControllerGlueValidator]:
        return ControllerGlueValidator

    @lru_cache(maxsize=1)
    def action_space(self) -> gym.spaces.Space:
        """
        Build the action space for the controller, etc.
        """
        action_space_dict = {}
        if isinstance(self._control_properties, list):
            action_spaces = [control_prop.create_space() for control_prop in self._control_properties]
            action_space_dict[self._key] = gym.spaces.tuple.Tuple(tuple(action_spaces))
        else:
            action_space_dict[self._key] = self._control_properties.create_space()

        return gym.spaces.Dict(action_space_dict)

    def apply_action(self, action: EnvSpaceUtil.sample_type, observation: EnvSpaceUtil.sample_type) -> None:
        """Apply the action for the controller, etc.

        Parameters
        ----------
        action : EnvSpaceUtil.sample_type
            The action that is to be applied at the controller level
        observation : EnvSpaceUtil.sample_type
            The current observable state by the agent (integration focus)
        """
        if isinstance(action, (tuple, list)):
            raise ValueError("Unexpected action of type tuple or list")
        control = action[self._key]
        self._controller.apply_control(control=control)

    def get_applied_control(self) -> OrderedDict:
        control_dict = OrderedDict()
        if not self._agent_removed:
            control_dict[self._key] = self._controller.get_validated_applied_control()
        else:
            control_dict[self._key] = self.action_space()[self._key].sample()

        return control_dict

    @lru_cache(maxsize=1)
    def observation_space(self) -> gym.spaces.Space:
        obs_space = gym.spaces.dict.Dict()

        obs_space.spaces['invalid'] = gym.spaces.Discrete(2)
        obs_space.spaces['control'] = self.action_space()[self._key]

        return obs_space

    def get_observation(self) -> OrderedDict:
        obs_dict = OrderedDict()

        obs_dict['invalid'] = 1 if self._agent_removed else 0
        obs_dict['invalid'] = 1 if obs_dict['invalid'] or not self.controller.valid else 0
        obs_dict['control'] = self.get_applied_control()[self._key]

        return obs_dict

    @lru_cache(maxsize=1)
    def get_unique_name(self) -> str:
        """Provies a unique name of the glue to differentiate it from other glues.
        """
        control_properties = self._control_properties
        if isinstance(control_properties, list):
            controller_names: typing.List[str] = []
            for controller in control_properties:
                controller_names.append(controller.name)
            unique_name = ''.join([controller_name.capitalize() for controller_name in controller_names])
        else:
            unique_name = control_properties.name.capitalize()

        return self.config.controller + "_" + unique_name

    @property
    def resolved_controller_class_name(self) -> str:
        """Class name of the internal controller."""
        return type(self._controller).__name__
