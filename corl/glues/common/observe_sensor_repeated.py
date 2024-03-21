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
import typing
from collections import OrderedDict
from functools import cached_property

import gymnasium

from corl.glues.base_glue import BaseAgentPlatformGlue, BaseAgentPlatformGlueValidator
from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.property import DictProp, RepeatedProp
from corl.simulators.common_platform_utils import get_sensor_by_name


class ObserveSensorRepeatedValidator(BaseAgentPlatformGlueValidator):
    """
    sensor: which sensor to find on the platform
    output_units: unit to convert the output data to
    max_len: the maximum length to allow the observation space to reach
    """

    sensor: str
    output_units: dict[str, str] = {}
    max_len: int = 10


class ObserveSensorRepeated(BaseAgentPlatformGlue):
    """Glue to observe a sensor that can offer a variable length output

    Configuration:
        sensor: string      String of sensor class, sensor base name, or the PluginLibrary group name
                            For example consider a platform which has "SimOrientationRateSensor" attached.
                            The following will all be equivalent:
                            - SimOrientationRateSensor
                            - BaseOrientationRateSensor
                            - Sensor_OrientationRate

                            Important: If multiple sensors are found an array will be returned (ex. Gload)

                            This will also be the name attached to the glue, which can then be used
    """

    class Fields:
        """
        Fields in this glue
        """

        DIRECT_OBSERVATION = "direct_observation"

    @staticmethod
    def get_validator() -> type[ObserveSensorRepeatedValidator]:
        return ObserveSensorRepeatedValidator

    # TODO: self._sensor.measurement_properties assumes child_space attribute
    def __init__(self, **kwargs) -> None:
        self.config: ObserveSensorRepeatedValidator
        super().__init__(**kwargs)
        self._sensor = get_sensor_by_name(self._platform, self.config.sensor)
        self._sensor_name: str = self.config.sensor
        self.out_units = self.config.output_units
        self.max_len = self.config.max_len

    def get_unique_name(self) -> str:
        """Class method that retrieves the unique name for the glue instance"""
        # TODO: This may result in ObserveSensor_Sensor_X, which is kinda ugly
        return f"ObserveSensorRepeated_{self._sensor_name}"

    def invalid_value(self) -> OrderedDict:
        """Return zeros when invalid"""
        arr: list[float] = []
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = arr
        return d

    @cached_property
    def observation_units(self):
        """Units of the sensors in this glue"""
        return self._observation_units

    @cached_property
    def observation_prop(self):
        tmp = self._sensor.measurement_properties
        assert isinstance(self._sensor.measurement_properties, RepeatedProp), "Unexpected measurement_properties type"
        child_space = tmp.child_space
        for prop_name, unit in self.out_units.items():
            child_space[prop_name] = child_space[prop_name].create_unit_converted_prop(unit)
        ret = RepeatedProp(child_space=child_space, max_len=self.max_len)
        return DictProp(spaces={self.Fields.DIRECT_OBSERVATION: ret})

    # TODO: Assumes s`elf._sensor.measurement_properties has child_space attribute
    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> OrderedDict:
        """Observation Values"""
        # NOTE: do not attempt to optimize this function by modifying data in place
        # you will mess up all other glues that call self._glue.get_observation
        # a copy is required here
        sensed_value = self._sensor.get_measurement()
        assert isinstance(sensed_value, list)
        tmp_sensed: list[dict[str, typing.Any]] = []
        append = tmp_sensed.append

        out_units = self.out_units
        for platform_data in sensed_value:
            tmp_row = {}
            for field_name, obs in platform_data.items():
                tmp_row[field_name] = obs.to(out_units[field_name]) if field_name in out_units else obs

            append(tmp_row)
            # we are going to clip the list to the max_len, so the obs is happy
            if len(tmp_sensed) == self.max_len:
                break

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = tmp_sensed

        return d

    @cached_property
    def action_space(self) -> gymnasium.spaces.Space | None:
        """No Actions"""
        return None

    def apply_action(
        self,  # noqa: PLR6301
        action: EnvSpaceUtil.sample_type,
        observation: EnvSpaceUtil.sample_type,
        action_space,
        obs_space,
        obs_units,
    ):
        """No Actions"""
        return
