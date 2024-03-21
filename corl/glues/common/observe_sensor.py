"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from collections import OrderedDict, abc
from functools import cached_property

import numpy as np

from corl.glues.base_glue import BaseAgentPlatformGlue, BaseAgentPlatformGlueValidator
from corl.libraries.property import DictProp
from corl.libraries.units import Quantity
from corl.simulators.common_platform_utils import get_sensor_by_name


class ObserveSensorValidator(BaseAgentPlatformGlueValidator):
    """
    output_units: unit to convert the output data to
    sensor: which sensor to find on the platform
    """

    sensor: str
    output_units: str | None = None


class ObserveSensor(BaseAgentPlatformGlue):
    """Glue to observe a sensor

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
    def get_validator() -> type[ObserveSensorValidator]:
        return ObserveSensorValidator

    def __init__(self, **kwargs) -> None:
        self.config: ObserveSensorValidator
        super().__init__(**kwargs)

        self._sensor = get_sensor_by_name(self._platform, self.config.sensor)
        self._sensor_name: str = self.config.sensor

    def get_unique_name(self):
        """Class method that retrieves the unique name for the glue instance"""
        # TODO: This may result in ObserveSensor_Sensor_X, which is kinda ugly
        if self.config.name not in (ObserveSensor.__name__, "sensor"):
            return self.config.name
        return f"Obs_{self._sensor_name}"

    # TODO: broken
    def invalid_value(self) -> OrderedDict:
        """Return zeros when invalid"""
        arr: list[float] = []
        if isinstance(self._sensor.measurement_properties.low, abc.Sequence):  # type: ignore
            arr.extend([0.0 for _ in self._sensor.measurement_properties.low])  # type: ignore
        elif np.isscalar(self._sensor.measurement_properties.low):  # type: ignore
            arr.append(0.0)
        else:
            raise RuntimeError("Expecting either array or scalar")

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.asarray(arr)
        return d

    @cached_property
    def observation_prop(self):
        tmp = self._sensor.measurement_properties
        if self.config.output_units:
            tmp = tmp.create_unit_converted_prop(self.config.output_units)
        return DictProp(spaces={self.Fields.DIRECT_OBSERVATION: tmp})

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units) -> OrderedDict:
        """Observation Values"""
        d: OrderedDict[str, Quantity] = OrderedDict()
        tmp = self._sensor.get_measurement()
        if self.config.output_units:
            assert isinstance(self.config.output_units, str)
            tmp = tmp.to(self.config.output_units)
        d[self.Fields.DIRECT_OBSERVATION] = tmp
        return d
