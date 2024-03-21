"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import abc
import typing
from functools import cached_property

from corl.libraries.units import Quantity
from corl.simulators.base_parts import BaseController, BasePlatformPartValidator, BaseSensor


class CommsValidator(BasePlatformPartValidator):
    """Validator for comms"""

    default_value: typing.Any | None = None


class Comms(BaseController, BaseSensor, abc.ABC):
    """Abstract platform part for storing/communicating data"""

    def __init__(self, parent_platform, config, property_class) -> None:
        self.config: CommsValidator
        super().__init__(parent_platform, config, property_class)

        self._properties_config = str(config["properties"])
        self._last_measurement = self.default_value
        self.set_invalid()

    @staticmethod
    def get_validator() -> type[BasePlatformPartValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return CommsValidator

    @cached_property
    def default_value(self) -> Quantity:
        """Return the default value for this part"""
        if self.config.default_value is not None:
            sample = self.config.default_value
            ret = self.measurement_properties.create_quantity(sample)
        else:
            space = self.measurement_properties
            ret = space.create_zero_sample()

        assert isinstance(ret, Quantity)
        return ret

    def _update_state(self, state):
        """Update state"""

    @abc.abstractmethod
    def _set_data(self, data: Quantity) -> None:
        """Write data to this part"""

    @abc.abstractmethod
    def _get_data(self) -> Quantity:
        """Read data from this part"""

    def apply_control(self, control) -> None:
        """
        The generic method to apply the control for this controller.

        Parameters
        ----------
        control
            The control to be executed by the controller
        """
        self.set_valid()
        assert isinstance(control, Quantity)
        self._set_data(control)

    def get_applied_control(self) -> Quantity:
        """
        Get the previously applied control that was given to the apply_control function
        Returns
        -------
        previously applied control that was given to the apply_control function
        """
        return self._get_data()

    def _calculate_measurement(self, state) -> Quantity:
        self._update_state(state)
        return self.get_applied_control()

    def calculate_and_cache_measurement(self, state):  # noqa: PLR6301
        """
        Calculates the measurement and caches the result in the _last_measurement variable

        Parameters
        ----------
        state: BaseSimulatorState
            The current state of the environment used to obtain the measurement
        """
        super().calculate_and_cache_measurement(state)


class TeamCommsValidator(CommsValidator):
    """Validator for team comms"""


class TeamComms(Comms):
    """Abstract platform part for communicating data between agents on the same team"""

    def __init__(self, parent_platform, config, property_class) -> None:
        self.config: TeamCommsValidator
        super().__init__(parent_platform, config, property_class)
        self.team = self.parent_platform.name[: [x.isdigit() for x in self.parent_platform.name].index(True)]

    @staticmethod
    def get_validator() -> type[BasePlatformPartValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return TeamCommsValidator
