"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import enum
import typing
from collections import OrderedDict, abc
from functools import lru_cache

import gym
import numpy as np
from pydantic import validator

from corl.glues.base_glue import BaseAgentPlatformGlue, BaseAgentPlatformGlueValidator
from corl.libraries.property import BoxProp, DiscreteProp, MultiBinary
from corl.libraries.units import Convert, GetStrFromUnit, GetUnitFromStr, NoneUnitType
from corl.simulators.common_platform_utils import get_sensor_by_name


class ObserveSensorValidator(BaseAgentPlatformGlueValidator):
    """
    output_units: unit to convert the output data to
    sensor: which sensor to find on the platform
    """
    sensor: str
    output_units: typing.Union[typing.Sequence[enum.Enum], typing.Sequence[typing.Sequence[enum.Enum]]] = []

    @validator('output_units', always=True, pre=True)
    def validate_output_units(cls, v, values):  # pylint: disable=no-self-argument, no-self-use
        """output_units validator"""
        units = []
        sensor_obj = get_sensor_by_name(values['platform'], values['sensor'])
        if not v:
            # Configuration does not provide output units
            if hasattr(sensor_obj.measurement_properties, 'unit'):
                # Sensor measurement properties has unit attribute, get the default unit
                for value in sensor_obj.measurement_properties.unit:
                    if not isinstance(value, str) and isinstance(value, abc.Sequence):
                        # list of lists 2D case
                        units.append([GetUnitFromStr(val).DEFAULT for val in value])
                    else:
                        # 1D case
                        units.append(GetUnitFromStr(value).DEFAULT)
            else:
                # Sensor measurement properties does not have unit attribute, use NoneUnitType.DEFAULT
                units.append(NoneUnitType.DEFAULT)
        else:
            # Configuration provides output units, convert string units to enum.Enum units
            # Assert that sensor has units
            assert hasattr(sensor_obj.measurement_properties, 'unit'), "Unexpected output_units when sensor does not have unit attribute"
            if isinstance(v, str):
                # Configuration units are a single string
                assert isinstance(v, str), "Unexpected input output_units type"
                assert len(sensor_obj.measurement_properties.unit) == 1, "Unexpected length of output_units"
                units.append(GetUnitFromStr(v))
            elif isinstance(v, abc.Sequence):
                # Configuration units are a list
                assert len(v) == len(sensor_obj.measurement_properties.unit), "Unexpected length of output_units"
                for i, value in enumerate(v):
                    if not isinstance(value, str) and isinstance(value, abc.Sequence):
                        # list of lists 2D case
                        assert len(value) == len(sensor_obj.measurement_properties.unit[i]), "Unexpected length of output_units"
                        sub_units = []
                        for j, val in enumerate(value):
                            assert isinstance(val, str), "Unexpected input output_units type"
                            assert isinstance(GetUnitFromStr(sensor_obj.measurement_properties.unit[i][j]),
                                              type(GetUnitFromStr(val))), "Unexpected unit dimension"
                            sub_units.append(GetUnitFromStr(val))
                        units.append(sub_units)
                    else:
                        # 1D case
                        assert isinstance(value, str), "Unexpected input output_units type"
                        assert isinstance(GetUnitFromStr(sensor_obj.measurement_properties.unit[i]),
                                          type(GetUnitFromStr(value))), "Unexpected unit dimension"
                        units.append(GetUnitFromStr(value))
            else:
                raise TypeError(f'Unexpected_input output_units type: {type(v).__name__}')
        return units


class ObserveSensor(BaseAgentPlatformGlue):
    """Glue to observe a sensor

    Configuration:
        sensor: string      String of sensor class, sensor base name, or the PluginLibrary group name
                            For example consider a platform which has "SimOrientationRateSensor" attached.
                            The following will all be equivalent:
                            - SimOrientationRateSensor
                            - BaseOrientationRateSensor
                            - Sensor_OrientationRate

                            Important: If multiple sensors are found an array will be returned (ex. Gload Learjet)

                            This will also be the name attached to the glue, which can then be used
    """

    # pylint: disable=too-few-public-methods
    class Fields:
        """
        Fields in this glue
        """
        DIRECT_OBSERVATION = "direct_observation"

    @property
    def get_validator(self) -> typing.Type[ObserveSensorValidator]:
        return ObserveSensorValidator

    def __init__(self, **kwargs) -> None:
        self.config: ObserveSensorValidator
        super().__init__(**kwargs)

        self._sensor = get_sensor_by_name(self._platform, self.config.sensor)
        self._sensor_name: str = self.config.sensor
        self.out_units = self.config.output_units

    @lru_cache(maxsize=1)
    def get_unique_name(self) -> str:
        """Class method that retreives the unique name for the glue instance
        """
        # TODO: This may result in ObserveSensor_Sensor_X, which is kinda ugly
        return "ObserveSensor_" + self._sensor_name

    # TODO: broken
    def invalid_value(self) -> OrderedDict:
        """Return zeros when invalid
        """
        arr: typing.List[float] = []
        if isinstance(self._sensor.measurement_properties.low, abc.Sequence):  # type: ignore
            for _ in enumerate(self._sensor.measurement_properties.low):  # type: ignore
                arr.append(0.0)
        elif np.isscalar(self._sensor.measurement_properties.low):  # type: ignore
            arr.append(0.0)
        else:
            raise RuntimeError("Expecting either array or scalar")

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.asarray(arr)
        return d

    @lru_cache(maxsize=1)
    def observation_units(self):
        """Units of the sensors in this glue
        """
        out_units = []
        for value in self.out_units:
            if isinstance(value, abc.Sequence):
                # list of lists 2D case
                sub_units = []
                for val in value:
                    sub_units.append(GetStrFromUnit(val))
                out_units.append(sub_units)
            else:
                # 1D case
                out_units.append(GetStrFromUnit(value))
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = out_units
        return d

    @lru_cache(maxsize=1)
    def observation_space(self) -> gym.spaces.Space:
        """Observation Space
        """
        d = gym.spaces.dict.Dict()
        if isinstance(self._sensor.measurement_properties, BoxProp):
            d.spaces[self.Fields.DIRECT_OBSERVATION] = self._sensor.measurement_properties.create_converted_space(self.out_units)
        elif isinstance(self._sensor.measurement_properties, (DiscreteProp)):
            d.spaces[self.Fields.DIRECT_OBSERVATION] = self._sensor.measurement_properties.create_space()
        elif isinstance(self._sensor.measurement_properties, (MultiBinary)):
            d.spaces[self.Fields.DIRECT_OBSERVATION] = self._sensor.measurement_properties.create_space()
        else:
            raise TypeError("Only supports {BoxProp.__name__}, {MultiBinary.__name__} and {DiscreteProp.__name__}")
        return d

    def get_observation(self) -> OrderedDict:
        """Observation Values
        """
        d = OrderedDict()
        if isinstance(self._sensor.measurement_properties, BoxProp):
            sensed_value = self._sensor.get_measurement()
            if isinstance(sensed_value, np.ndarray):
                sensed_value = sensed_value.tolist()
            else:
                sensed_value = list(sensed_value)
            for indx, value in enumerate(sensed_value):
                if isinstance(value, abc.Sequence):
                    # list of lists 2D case
                    for indx1, value1 in enumerate(value):
                        in_unit2d = self._sensor.measurement_properties.unit[indx][indx1]
                        assert isinstance(in_unit2d, str)
                        assert isinstance(self.out_units, abc.Sequence)
                        out_unit1 = self.out_units[indx]
                        assert isinstance(out_unit1, abc.Sequence)
                        out_unit2d = out_unit1[indx1]
                        assert isinstance(out_unit2d, enum.Enum)
                        assert isinstance(sensed_value, abc.Sequence)
                        sensed_value1 = sensed_value[indx]
                        assert isinstance(sensed_value1, list)
                        sensed_value1[indx1] = Convert(value1, in_unit2d, out_unit2d)
                else:
                    # 1D case
                    in_unit = self._sensor.measurement_properties.unit[indx]
                    assert isinstance(in_unit, str)
                    out_unit = self.out_units[indx]
                    assert isinstance(out_unit, enum.Enum)
                    assert isinstance(sensed_value, list)
                    sensed_value[indx] = Convert(value, in_unit, out_unit)
            d[self.Fields.DIRECT_OBSERVATION] = np.array(sensed_value, dtype=np.float32)
        elif isinstance(self._sensor.measurement_properties, (DiscreteProp)):
            sensed_value_discrete: float = self._sensor.get_measurement()[0]
            d[self.Fields.DIRECT_OBSERVATION] = np.array(sensed_value_discrete, dtype=np.int32)
        elif isinstance(self._sensor.measurement_properties, (MultiBinary)):
            sensed_value_multi_binary = self._sensor.get_measurement()
            d[self.Fields.DIRECT_OBSERVATION] = sensed_value_multi_binary  # type: ignore
        else:
            raise TypeError("Only supports {BoxProp.__name__}, {MultiBinary.__name__} and {DiscreteProp.__name__}")

        return d

    @lru_cache(maxsize=1)
    def action_space(self) -> gym.spaces.Space:
        """No Actions
        """
        return None

    def apply_action(self, action, observation):
        """No Actions
        """
        return None
