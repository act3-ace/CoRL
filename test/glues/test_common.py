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
import gym
import numpy as np
from pydantic import ValidationError
import pytest

import corl.glues.common.unit_vector_glue
from corl.libraries.functor import FunctorWrapper

from corl.glues.base_glue import BaseAgentGlue
from corl.libraries.property import BoxProp
from corl.simulators.base_parts import BaseSensor
from corl.simulators.base_platform import BasePlatform
from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.openai_gym.gym_simulator import OpenAIGymSimulator, OpenAiGymPlatform
from corl.simulators.openai_gym.gym_available_platforms import OpenAIGymAvailablePlatformTypes
import mock


class TestSensor1DArray(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=BoxProp)
    def _calculate_measurement(self, state):
        return np.array([50.0]).astype(np.float32)

PluginLibrary.AddClassToGroup(
    TestSensor1DArray, "Sensor_Test1DArray", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


class TestSensor2DArray(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=BoxProp)
    def _calculate_measurement(self, state):
        return np.array([50.0, 25.0]).astype(np.float32)

PluginLibrary.AddClassToGroup(
    TestSensor2DArray, "Sensor_Test2DArray", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


class TestSensor3DArray(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=BoxProp)
    def _calculate_measurement(self, state):
        return np.array([50.0, 25.0, 12.5]).astype(np.float32)

PluginLibrary.AddClassToGroup(
    TestSensor3DArray, "Sensor_Test3DArray", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


class TestSensorAngle(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=BoxProp)
    def _calculate_measurement(self, state):
        return np.array([np.pi / 4.0]).astype(np.float32)

PluginLibrary.AddClassToGroup(
    TestSensorAngle, "TestSensorAngle", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


@pytest.fixture(name="platform_name")
def get_platform_name():
    return  "test_platform"

@pytest.fixture(name="parts_list")
def get_parts_list():
    return  [(TestSensor1DArray, {
        "properties": {
            "name": "TestSensor1DArray",
            "description":"TestSensor1DArray",
            "low": [0.0],
            "high":[100.0],
            "unit":["None"]
            }
        }),
        (TestSensor2DArray, {
            "properties": {
                "name": "TestSensor2DArray",
                "description":"TestSensor2DArray",
                "low": [0.0, 0.0],
                "high":[100.0, 50.0],
                "unit":["None", "None"]
            }
        }),
        (TestSensor3DArray, {
            "properties": {
                "name": "TestSensor3DArray",
                "description":"TestSensor3DArray",
                "low": [0.0, 0.0, 0.0],
                "high":[100.0, 50.0, 25.0],
                "unit":["None", "None", "None"]
            }
        }),
        (TestSensorAngle, {
            "properties": {
                "name": "TestSensorAngle",
                "description":"TestSensorAngle",
                "low": [0.0],
                "high":[2 * np.pi],
                "unit":["rad"]
            }
        }) ]

@pytest.fixture(name="platform")
def get_platform(platform_name, parts_list):
    return OpenAiGymPlatform(platform_name=platform_name, platform=gym.make("CartPole-v1"), parts_list=parts_list)

@pytest.fixture(name="platform_config")
def get_platform_config():
    return { "platform_class": "corl.simulators.openai_gym.gym_simulator.OpenAiGymInclusivePartsPlatform"
    }


def unit_vec_standardnorm(sensor_name: str, array_size: int):
    return {
        "functor": "corl.glues.common.unit_vector_glue.UnitVectorGlue",
        "config": {
            "normalization": {
                "enabled": True,
                "mu": [0] * array_size,
                "sigma": [1] * array_size,
            }
        },
        "wrapped": {
            "functor": "corl.glues.common.observe_sensor.ObserveSensor",
            "config": {
                "sensor": sensor_name,
            }
        },
    }

def trig_values_glue(sensor_name: str):
    return {
        "functor": "corl.glues.common.trig_values.TrigValues",
        "wrapped": {
            "functor": "corl.glues.common.observe_sensor.ObserveSensor",
            "config":{
                "sensor": sensor_name,
            },
        }
    }

def mag_glue(sensor_name: str):
    return {
        "functor": "corl.glues.common.magnitude.MagnitudeGlue",
        "wrapped": {
            "functor": "corl.glues.common.observe_sensor.ObserveSensor",
            "config":{
                "sensor": sensor_name,
            },
        }
    }

@pytest.mark.parametrize("config, normalized_obs_space, normalized_observation",
    [
        pytest.param(
            unit_vec_standardnorm(sensor_name="Sensor_Test1DArray", array_size=1),
            gym.spaces.Dict({
                "unit_vec": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
            }),
            OrderedDict({"unit_vec": np.array([1.0], dtype=np.float32)}),
            id="1d_unit_vector_std_norm"),
        pytest.param(
            unit_vec_standardnorm(sensor_name="Sensor_Test2DArray", array_size=2),
            gym.spaces.Dict({
                "unit_vec": gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])),
            }),
            OrderedDict({"unit_vec": np.array([0.8944272, 0.4472136], dtype=np.float32)}),
            id="2d_unit_vector_std_norm"),
        pytest.param(
            unit_vec_standardnorm(sensor_name="Sensor_Test3DArray", array_size=3),
            gym.spaces.Dict({
                "unit_vec": gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0])),
            }),
            OrderedDict({"unit_vec": np.array([0.8728716, 0.4364358, 0.2182179], dtype=np.float32)}),
            id="3d_unit_vector_std_norm"),
        pytest.param(
            trig_values_glue(sensor_name="TestSensorAngle"),
            gym.spaces.Dict({
                "cos": gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0])),
                "sin": gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0])),
            }),
            OrderedDict({
                "cos": np.array([np.cos(np.pi / 4)], dtype=np.float32),
                "sin": np.array([np.cos(np.pi / 4)], dtype=np.float32)
            }),
            id="trig_values_linear"),
        pytest.param(
            mag_glue(sensor_name="Sensor_Test2DArray"),
            gym.spaces.Dict({
                "mag": gym.spaces.Box(low=np.array([0.0]), high=np.array([111.8034])),
            }),
            OrderedDict({
                "mag": np.array([55.9017], dtype=np.float32),
            }),
            id="mag_values_linear"),
    ])
def test_obs_glue(config, normalized_obs_space, normalized_observation, platform):
    functor = FunctorWrapper(**config)
    created_glue = functor.create_functor_object(platform=platform, agent_name="test_agent")

    created_glue.glue()._sensor.calculate_and_cache_measurement({})

    assert normalized_obs_space == created_glue.observation_space()
    np.testing.assert_equal(normalized_observation, created_glue.get_observation({}, {}, {}))
    assert None == created_glue.action_space()
    assert None == created_glue.apply_action({}, {}, {}, {}, {})
