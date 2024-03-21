"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from collections import OrderedDict

import gymnasium
from pydantic import field_validator

from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from corl.libraries.property import BoxProp
from corl.libraries.units import Quantity, corl_get_ureg
from corl.simulators.base_simulator import BaseSimulatorState
from corl.simulators.common_platform_utils import get_platform_by_name, get_sensor_by_name


class SensorBoundsCheckDoneValidator(DoneFuncBaseValidator):
    """Initialize an SensorBoundsCheckDone

    Parameters
    ----------
    min_value : float
        The minimum allowed value of this sensor
    max_value : float
        The maximum allowed value of this sensor
    sensor_name : str
        The name of the sensor to check
    """

    min_value: Quantity
    max_value: Quantity
    sensor_name: str

    @field_validator("max_value")
    def min_max_consistent(cls, v, values):
        """Validate that the maximum is bigger than the minimum."""
        if v.units != values.data["min_value"].units:
            raise ValueError(f'Inconsistent units for min and max: {values.data["min_value"].units}, {v.units}')
        if v.m <= values.data["min_value"].m:
            raise ValueError(f'Minimum bound {values.data["min_value"].m} exceeds maximum bound {v.m}')
        return v


class SensorBoundsCheckDone(DoneFuncBase):
    """
    Checks to see if a specified state parameter is within bounds
    """

    # True means that clients should pass ValueWithUnits rather than evaluated value
    REQUIRED_UNITS = {"min_value": True, "max_value": True, "sensor_name": None}

    def __init__(self, **kwargs) -> None:
        self.config: SensorBoundsCheckDoneValidator
        super().__init__(**kwargs)
        if self.config.name == type(self).__name__:
            self.config.name = f"{self.config.sensor_name}_Done"

    @staticmethod
    def get_validator() -> type[SensorBoundsCheckDoneValidator]:
        """Returns the validator for this done condition"""
        return SensorBoundsCheckDoneValidator

    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: BaseSimulatorState,
        observation_space: gymnasium.Space,
        observation_units: OrderedDict,
    ) -> bool:
        # Find Target Platform
        platform = get_platform_by_name(next_state, self.platform, allow_invalid=True)

        # platform does not exist
        if platform is None:
            return False

        # Find target Sensor
        sensor = get_sensor_by_name(platform, self.config.sensor_name)
        if not isinstance(sensor.measurement_properties, BoxProp):
            raise TypeError(f"Can only do bounds checking on BoxProp, received {type(sensor.measurement_properties).__name__}")

        # Get measured value
        measurement = sensor.get_measurement()
        assert len(measurement.m)
        if len(measurement.m) != 1:
            raise ValueError("Sensor measurement has more than one element")
        measured_value = corl_get_ureg().Quantity(value=measurement.m[0], units=sensor.measurement_properties.get_units())
        converted_value = measured_value.to(str(self.config.min_value.u))

        # Determine if done
        done = converted_value.m < self.config.min_value.m or self.config.max_value.m < converted_value.m

        if done:
            next_state.episode_state[self.platform][self.name] = DoneStatusCodes.LOSE

        return bool(done)
