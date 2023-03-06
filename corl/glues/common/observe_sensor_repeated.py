"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------
ObserveSensorRepeated Glue
"""
import enum
import typing
from collections import OrderedDict
from functools import lru_cache

import gym
import numpy as np
from pydantic import validator
from ray.rllib.utils.spaces.repeated import Repeated

from corl.glues.base_glue import BaseAgentPlatformGlue, BaseAgentPlatformGlueValidator
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.property import BoxProp, DiscreteProp, MultiBinary, RepeatedProp
from corl.libraries.units import Convert, GetStrFromUnit, GetUnitFromStr, NoneUnitType
from corl.simulators.common_platform_utils import get_sensor_by_name


class ObserveSensorRepeatedValidator(BaseAgentPlatformGlueValidator):
    """
    sensor: which sensor to find on the platform
    output_units: unit to convert the output data to
    max_len: the maximum length to allow the observation space to reach
    enable_clip: enables clipping for spaces that support cliping
    """
    sensor: str
    output_units: typing.Dict[str, enum.Enum] = {}
    max_len: int = 10
    enable_clip: bool = False

    @validator('output_units', always=True, pre=True)
    def validate_output_units(cls, v, values):  # pylint: disable=no-self-argument
        """output_units validator"""
        sensor_obj = get_sensor_by_name(values['platform'], values['sensor'])
        child_space = sensor_obj.measurement_properties.child_space
        units = {}
        if v is not None:
            # Configuration provides output units, convert string units to enum.Enum units
            # Assert that input configuration is a dict
            assert isinstance(v, dict), "Configuration output_units must be a dict"
            for field_name, value in v.items():
                if field_name not in child_space:
                    raise RuntimeError(
                        f"output units were provided for repeated field {field_name}, but "
                        f"that field is not in this repeated space which has keys {child_space.keys()}"
                    )
                prop = child_space[field_name]
                if not isinstance(prop, (DiscreteProp, MultiBinary)):
                    assert isinstance(prop, BoxProp), "Unexpected field_space type"
                    if np.isscalar(prop.high) or len(prop.high) == 1:
                        if value is not None and not isinstance(value, str):
                            raise RuntimeError("output units must be provided as a scalar")
                    else:
                        raise RuntimeError("repeated field space unit conversions with lists of output types not currently implemented")
                    assert isinstance(GetUnitFromStr(prop.unit[0]), type(GetUnitFromStr(value))), "Unexpected unit dimension"
                    units[field_name] = GetUnitFromStr(value)
        # Fill in any output units that were not provided with values from the sensor
        for field_name, prop in child_space.items():
            if field_name in units:
                # This field name already provided so skip
                continue
            if hasattr(prop, 'unit'):
                # Sensor measurement properties has unit attribute, get the default unit
                if isinstance(prop.unit, str):
                    units[field_name] = GetUnitFromStr(prop.unit).DEFAULT
                elif isinstance(prop.unit, list):
                    units[field_name] = GetUnitFromStr(prop.unit[0]).DEFAULT
                else:
                    raise RuntimeError("Unexpected property unit type")
            else:
                # Sensor measurement properties does not have unit attribute, use NoneUnitType.DEFAULT
                units[field_name] = NoneUnitType.DEFAULT
        return units


class ObserveSensorRepeated(BaseAgentPlatformGlue):
    """Glue to observe a sensor that can offer a variable length output

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
    def get_validator(self) -> typing.Type[ObserveSensorRepeatedValidator]:
        return ObserveSensorRepeatedValidator

    # TODO: self._sensor.measurement_properties assumes child_space attribute
    def __init__(self, **kwargs) -> None:
        self.config: ObserveSensorRepeatedValidator
        super().__init__(**kwargs)
        self._sensor = get_sensor_by_name(self._platform, self.config.sensor)
        self._sensor_name: str = self.config.sensor
        self.out_units = self.config.output_units
        self.max_len = self.config.max_len

    @lru_cache(maxsize=1)
    def get_unique_name(self) -> str:
        """Class method that retreives the unique name for the glue instance
        """
        # TODO: This may result in ObserveSensor_Sensor_X, which is kinda ugly
        return "ObserveSensorRepeated_" + self._sensor_name

    def invalid_value(self) -> OrderedDict:
        """Return zeros when invalid
        """
        arr: typing.List[float] = []
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = arr
        return d

    @lru_cache(maxsize=1)
    def observation_units(self):
        """Units of the sensors in this glue
        """
        out_dict = {}
        for field_name, field_unit in self.out_units.items():
            out_dict[field_name] = GetStrFromUnit(field_unit)

        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = out_dict
        return d

    # TODO: Assumes self._sensor.measurement_properties has attribute child_space
    @lru_cache(maxsize=1)
    def observation_space(self):
        """Observation Space
        """

        d = gym.spaces.dict.Dict()
        child_space = gym.spaces.dict.Dict()
        assert isinstance(self._sensor.measurement_properties, RepeatedProp), "Unexpected measurement_properties type"
        child_space_measure = self._sensor.measurement_properties.child_space
        for field_name, field_meas in child_space_measure.items():
            if not isinstance(field_meas, (DiscreteProp, MultiBinary)):
                assert isinstance(field_meas, BoxProp), "Unexpected field_meas type"
                if field_name not in self.out_units:
                    child_space.spaces[field_name] = field_meas.create_space()
                elif isinstance(self.out_units[field_name], list):
                    units = self.out_units[field_name]
                    assert isinstance(units, list), "Unexpected units type"
                    child_space.spaces[field_name] = field_meas.create_converted_space(units)
                else:
                    child_space.spaces[field_name] = field_meas.create_converted_space([self.out_units[field_name]] * len(field_meas.low))
            else:
                child_space.spaces[field_name] = field_meas.create_space()

        d.spaces[self.Fields.DIRECT_OBSERVATION] = Repeated(
            child_space=child_space,
            max_len=self.config.max_len,
        )

        return d

    # TODO: Assumes self._sensor.measurement_properties has child_space attribute
    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> OrderedDict:
        """Observation Values
        """
        # NOTE: do not attempt to optimize this function by modifying data in place
        # you will mess up all other glues that call self._glue.get_observation
        # a copy is required here
        sensed_value = self._sensor.get_measurement()
        tmp_sensed: typing.List[typing.Dict[str, typing.Any]] = []
        append = tmp_sensed.append
        assert isinstance(self._sensor.measurement_properties, RepeatedProp), "Unexpected measurement_properties type"
        m_props = self._sensor.measurement_properties.child_space
        out_units = self.out_units
        obs_child_space = self.observation_space()[self.Fields.DIRECT_OBSERVATION].child_space
        for platform_data in sensed_value:
            tmp_row = {}
            for field_name, obs in platform_data.items():
                tmp_row[field_name] = obs
                if field_name in out_units:
                    prop = m_props[field_name]
                    if isinstance(prop, BoxProp):
                        old_unit = prop.unit[0]
                        assert isinstance(old_unit, str)
                        unit_class = GetUnitFromStr(old_unit)
                        if unit_class != NoneUnitType.NoneUnit:
                            tmp_row[field_name] = Convert(obs, old_unit, out_units[field_name])
                if self.config.enable_clip:
                    field_space = obs_child_space.spaces[field_name]
                    if isinstance(field_space, gym.spaces.Box):
                        tmp_row[field_name] = np.clip(obs, field_space.low, field_space.high)
            append(tmp_row)
            # we are going to clip the list to the max_len, so the obs is happy
            if len(tmp_sensed) == self.max_len:
                break

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = tmp_sensed

        return d

    @lru_cache(maxsize=1)
    def action_space(self):
        """No Actions
        """
        return None

    def apply_action(self, action: EnvSpaceUtil.sample_type, observation: EnvSpaceUtil.sample_type, action_space, obs_space, obs_units):
        """No Actions
        """
        return None
