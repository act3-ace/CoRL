"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Sensors for OpenAIGymSimulator
"""
import typing

import numpy as np
from numpy_ringbuffer import RingBuffer

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp, RepeatedProp
from corl.simulators.base_parts import BaseSensor
from corl.simulators.openai_gym.gym_available_platforms import OpenAIGymAvailablePlatformTypes
from corl.simulators.openai_gym.gym_simulator import OpenAIGymSimulator


class OpenAiGymStateSensor(BaseSensor):
    """
    Sensor that reports the observation of a given platform gym environment
    """

    def __init__(self, parent_platform, config=None):

        obs_space = parent_platform.observation_space

        class GymSensorProp(BoxProp):
            """
            GymSensorProp can be updated via config and valdidate by pydantic
            """
            name: str = "GymStateSensor"
            low: typing.List[float] = obs_space.low.tolist()
            high: typing.List[float] = obs_space.high.tolist()
            unit: typing.List[str] = ["None"] * len(obs_space.low)
            description: str = "Gym Space"

        super().__init__(parent_platform=parent_platform, config=config, property_class=GymSensorProp)

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return {"state_sensor"}

    def _calculate_measurement(self, state):
        return state.obs[self.parent_platform.name]


PluginLibrary.AddClassToGroup(
    OpenAiGymStateSensor, "Sensor_State", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


class OpenAiGymRepeatedStateSensor(BaseSensor):
    """
    Sensor that reports the observation of a given platform gym environment
    """

    def __init__(self, parent_platform, config=None):

        obs_space = parent_platform.observation_space

        class GymSensorProp(BoxProp):
            """
            GymSensorProp can be updated via config and valdidate by pydantic
            """
            name: str = "GymStateSensor"
            low: typing.List[float] = obs_space.low.tolist()
            high: typing.List[float] = obs_space.high.tolist()
            unit: typing.List[str] = ["None"] * len(obs_space.low)
            description: str = "Gym Space"

        class GymSensorRepeatedProp(RepeatedProp):
            """
            GymSensorProp can be updated via config and valdidate by pydantic
            """
            name: str = "GymStateRepeatedSensor"
            max_len: int = 10
            child_space: typing.Dict[str, GymSensorProp] = {"GymState": GymSensorProp()}  # type: ignore
            description: str = "Gym Space Repeated"

        super().__init__(config=config, parent_platform=parent_platform, property_class=GymSensorRepeatedProp)

        self.obs_buffer = RingBuffer(capacity=self.measurement_properties.max_len, dtype=np.ndarray)

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return {"state_sensor"}

    def _calculate_measurement(self, state):
        self.obs_buffer.append(state.obs[self.parent_platform.name])
        ret = map(lambda x: {"GymState": x}, self.obs_buffer)
        return list(ret)


PluginLibrary.AddClassToGroup(
    OpenAiGymRepeatedStateSensor,
    "Sensor_RepeatedState", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)
