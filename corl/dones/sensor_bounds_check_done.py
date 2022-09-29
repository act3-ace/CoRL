"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import typing
from collections import OrderedDict

from pydantic import validator

from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from corl.libraries.environment_dict import DoneDict
from corl.libraries.property import BoxProp
from corl.libraries.state_dict import StateDict
from corl.libraries.units import NoneUnitType, ValueWithUnits
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
    min_value: ValueWithUnits
    max_value: ValueWithUnits
    sensor_name: str

    @validator('max_value')
    def min_max_consistent(cls, v, values):
        """Validate that the maximum is bigger than the minimum."""
        if 'min_value' not in values:
            # min_value failed, so let that error message be provided
            return v
        if v.units != values['min_value'].units:
            raise ValueError(f'Inconsistent units for min and max: {values["min_value"].units}, {v.units}')
        if v.value <= values['min_value'].value:
            raise ValueError(f'Minimum bound {values["min_value"].value} exceeds maximum bound {v.value}')
        return v


class SensorBoundsCheckDone(DoneFuncBase):
    """
    Checks to see if a specified state parameter is within bounds
    """

    # True means that clients should pass ValueWithUnits rather than evaluated value
    REQUIRED_UNITS = {'min_value': True, 'max_value': True, 'sensor_name': NoneUnitType.NoneUnit}

    def __init__(self, **kwargs) -> None:
        self.config: SensorBoundsCheckDoneValidator
        super().__init__(**kwargs)
        if self.config.name == type(self).__name__:
            self.config.name = self.config.sensor_name + '_Done'

    @property
    def get_validator(self) -> typing.Type[SensorBoundsCheckDoneValidator]:
        """Returns the validator for this done condition"""
        return SensorBoundsCheckDoneValidator

    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> DoneDict:

        done = DoneDict()

        # Find Target Platform
        platform = get_platform_by_name(next_state, self.platform, allow_invalid=True)

        # platform does not exist
        if platform is None:
            done[self.platform] = False
            return done

        # Find target Sensor
        sensor = get_sensor_by_name(platform, self.config.sensor_name)
        if not isinstance(sensor.measurement_properties, BoxProp):
            raise TypeError(f'Can only do bounds checking on BoxProp, received {type(sensor.measurement_properties).__name__}')

        # Get measured value
        measurement = sensor.get_measurement()
        if len(measurement) != 1:
            raise ValueError("Sensor measurement has more than one element")
        measured_value = ValueWithUnits(value=measurement[0], units=sensor.measurement_properties.unit[0])
        converted_value = measured_value.as_units(self.config.min_value.units)

        # Determine if done
        done[self.platform] = converted_value < self.config.min_value.value or self.config.max_value.value < converted_value

        if done[self.platform]:
            next_state.episode_state[self.platform][self.name] = DoneStatusCodes.LOSE

        self._set_all_done(done)

        return done
