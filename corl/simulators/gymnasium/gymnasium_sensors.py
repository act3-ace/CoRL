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
# mypy really does not like the dynamic properties and dies on them,
# this file must be excluded from linting
# mypy: ignore-errors


import numpy as np
from numpy_ringbuffer import RingBuffer

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp, RepeatedProp
from corl.libraries.units import Quantity, corl_get_ureg
from corl.simulators.base_parts import BaseSensor
from corl.simulators.gymnasium.gymnasium_available_platforms import GymnasiumAvailablePlatformTypeMain
from corl.simulators.gymnasium.gymnasium_simulator import GymnasiumSimulator


class GymnasiumSensorProp(BoxProp):
    name: str = "GymnasiumStateSensor"
    unit: str = "dimensionless"
    description: str = "Gymnasium Space"


def create_gym_state_sensor_prop(obs_space) -> type[GymnasiumSensorProp]:
    class TmpGymnasiumSensorProp(GymnasiumSensorProp):  # type: ignore
        low: list[float] = obs_space.low.tolist()
        high: list[float] = obs_space.high.tolist()

    return TmpGymnasiumSensorProp


class GymnasiumStateSensor(BaseSensor):
    """
    Sensor that reports the observation of a given platform gymnasium environment
    """

    def __init__(self, parent_platform, config=None) -> None:
        obs_space = parent_platform.observation_space

        state_sensor_prop = create_gym_state_sensor_prop(obs_space=obs_space)

        super().__init__(parent_platform=parent_platform, config=config, property_class=state_sensor_prop)

    def _calculate_measurement(self, state) -> Quantity:
        return corl_get_ureg().Quantity(state.obs[self.parent_platform.name].astype("float32"), "dimensionless")


PluginLibrary.AddClassToGroup(
    GymnasiumStateSensor, "Sensor_State", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


class GymnasiumSensorRepeatedProp(RepeatedProp):
    """
    GymnasiumSensorProp can be updated via config and valdidate by pydantic
    """

    name: str = "GymnasiumStateRepeatedSensor"
    max_len: int = 10
    # child_space: dict[str, GymnasiumSensorProp]
    description: str = "Gymnasium Space Repeated"


def create_gym_state_repeated_sensor_prop(child_prop) -> type[GymnasiumSensorRepeatedProp]:
    class tmpGymnasiumSensorRepeatedProp(GymnasiumSensorRepeatedProp):
        child_space: dict[str, GymnasiumSensorProp] = {"GymnasiumState": child_prop()}

    return tmpGymnasiumSensorRepeatedProp


class GymnasiumRepeatedStateSensor(BaseSensor):
    """
    Sensor that reports the observation of a given platform gymnasium environment
    """

    def __init__(self, parent_platform, config=None) -> None:
        obs_space = parent_platform.observation_space

        state_sensor_prop = create_gym_state_sensor_prop(obs_space=obs_space)

        repeated_state_sensor_prop = create_gym_state_repeated_sensor_prop(state_sensor_prop)

        self.measurement_properties: RepeatedProp
        super().__init__(config=config, parent_platform=parent_platform, property_class=repeated_state_sensor_prop)

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
