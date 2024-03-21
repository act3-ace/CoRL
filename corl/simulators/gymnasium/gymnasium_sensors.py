"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Sensors for GymnasiumSimulator
"""

import numpy as np
from numpy_ringbuffer import RingBuffer

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp, RepeatedProp
from corl.libraries.units import Quantity, corl_get_ureg
from corl.simulators.base_parts import BaseSensor
from corl.simulators.gymnasium.gymnasium_available_platforms import GymnasiumAvailablePlatformTypeMain
from corl.simulators.gymnasium.gymnasium_simulator import GymnasiumSimulator


class GymnasiumStateSensor(BaseSensor):
    """
    Sensor that reports the observation of a given platform gymnasium environment
    """

    def __init__(self, parent_platform, config=None) -> None:
        obs_space = parent_platform.observation_space

        class GymnasiumSensorProp(BoxProp):
            """
            GymnasiumSensorProp can be updated via config and valdidate by pydantic
            """

            name: str = "GymnasiumStateSensor"
            low: list[float] = obs_space.low.tolist()
            high: list[float] = obs_space.high.tolist()
            unit: str = "dimensionless"
            description: str = "Gymnasium Space"

        super().__init__(parent_platform=parent_platform, config=config, property_class=GymnasiumSensorProp)

    def _calculate_measurement(self, state) -> Quantity:
        return corl_get_ureg().Quantity(state.obs[self.parent_platform.name].astype("float32"), "dimensionless")


PluginLibrary.AddClassToGroup(
    GymnasiumStateSensor, "Sensor_State", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


class GymnasiumRepeatedStateSensor(BaseSensor):
    """
    Sensor that reports the observation of a given platform gymnasium environment
    """

    def __init__(self, parent_platform, config=None) -> None:
        obs_space = parent_platform.observation_space

        class GymnasiumSensorProp(BoxProp):
            """
            GymnasiumSensorProp can be updated via config and valdidate by pydantic
            """

            name: str = "GymnasiumStateSensor"
            low: list[float] = obs_space.low.tolist()
            high: list[float] = obs_space.high.tolist()
            unit: str = "dimensionless"
            description: str = "Gymnasium Space"

        class GymnasiumSensorRepeatedProp(RepeatedProp):
            """
            GymnasiumSensorProp can be updated via config and valdidate by pydantic
            """

            name: str = "GymnasiumStateRepeatedSensor"
            max_len: int = 10
            child_space: dict[str, GymnasiumSensorProp] = {"GymnasiumState": GymnasiumSensorProp()}  # type: ignore
            description: str = "Gymnasium Space Repeated"

        self.measurement_properties: RepeatedProp
        super().__init__(config=config, parent_platform=parent_platform, property_class=GymnasiumSensorRepeatedProp)

        self.obs_buffer = RingBuffer(capacity=self.measurement_properties.max_len, dtype=np.ndarray)

    def _calculate_measurement(self, state):
        self.obs_buffer.append(corl_get_ureg().Quantity(state.obs[self.parent_platform.name].astype("float32"), "dimensionless"))
        ret = ({"GymnasiumState": x} for x in self.obs_buffer)
        return list(ret)


PluginLibrary.AddClassToGroup(
    GymnasiumRepeatedStateSensor,
    "Sensor_RepeatedState",
    {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain},
)
