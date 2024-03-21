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
from collections import OrderedDict
from functools import cached_property

from corl.glues.base_glue import BaseAgentControllerGlue, BaseAgentPlatformGlueValidator
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.property import DictProp
from corl.libraries.units import Quantity, corl_get_ureg
from corl.simulators.base_parts import BaseController
from corl.simulators.common_platform_utils import get_controller_by_name

code_logger = logging.getLogger("code_timing")


class ControllerGlueValidator(BaseAgentPlatformGlueValidator):
    """
    controller: Which controller to get from parent platform
    """

    controller: str


class ControllerGlue(BaseAgentControllerGlue):
    """
    This simple glue class wraps a controller and creates an action space based on the controller
    This class has no observation space or observations
    """

    def __init__(self, **kwargs) -> None:
        self.config: ControllerGlueValidator
        super().__init__(**kwargs)

        self._controller = get_controller_by_name(self._platform, self.config.controller)
        self._controller_name: str = self.config.controller
        self._key = self._controller.control_properties.name
        self._control_properties = self._controller.control_properties

        control_properties = self._control_properties
        if isinstance(control_properties, list):
            controller_names: list[str] = [controller.name for controller in control_properties]
            unique_name = "".join([controller_name.capitalize() for controller_name in controller_names])
        else:
            assert control_properties.name is not None
            unique_name = control_properties.name.capitalize()

        self._uname = f"{self.config.controller}_{unique_name}"

    @property
    def controller(self) -> BaseController:
        """Returns controller

        Returns
        -------
        BaseController
            The controller for this glue
        """
        return self._controller

    @staticmethod
    def get_validator() -> type[ControllerGlueValidator]:
        return ControllerGlueValidator

    @cached_property
    def action_prop(self):
        return self._control_properties

    @cached_property
    def observation_prop(self):
        tmp = {"control": self._control_properties}
        return DictProp(spaces=tmp)

    def apply_action(
        self, action: EnvSpaceUtil.sample_type, observation: EnvSpaceUtil.sample_type, action_space, obs_space, obs_units
    ) -> None:
        """Apply the action for the controllers

        Parameters
        ----------
        action : EnvSpaceUtil.sample_type
            The action that is to be applied at the controller level
        observation : EnvSpaceUtil.sample_type
            The current observable state by the agent (integration focus)
        """
        assert isinstance(action, Quantity) or (
            isinstance(action, dict) and all(isinstance(action_value, Quantity) for action_value in action.values())
        )
        self._controller.apply_control(control=action)

    def get_applied_control(self):
        if self._agent_removed:
            assert self.action_space is not None
            return corl_get_ureg().Quantity(self.action_space.sample(), self.action_prop.get_units())
        return self._controller.get_validated_applied_control()

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> OrderedDict:
        obs_dict = OrderedDict()
        obs_dict["control"] = self.get_applied_control()
        return obs_dict

    def get_unique_name(self):
        """Provides a unique name of the glue to differentiate it from other glues."""
        return self._uname

    @property
    def resolved_controller_class_name(self) -> str:
        """Class name of the internal controller."""
        return type(self._controller).__name__


class ControllerGlueLegacy(ControllerGlue):
    """
    Version of ControllerGlue that matches the corl2 action space, where
    action space was a Dict
    """

    @cached_property
    def action_prop(self):
        tmp = {self._key: super().action_prop}
        return DictProp(spaces=tmp)

    def apply_action(self, action, observation, action_space, obs_space, obs_units):
        control = action[self._key]
        return super().apply_action(control, observation, action_space, obs_space, obs_units)
