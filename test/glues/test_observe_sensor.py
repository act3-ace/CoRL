"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import gym
import numpy as np
from pydantic import ValidationError
import pytest
import typing
import math

from corl.glues.common.observe_sensor import ObserveSensor
from corl.libraries.property import BoxProp
from corl.simulators.base_parts import BaseSensor
from corl.simulators.base_platform import BasePlatform
from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.openai_gym.gym_simulator import OpenAIGymSimulator
from corl.simulators.openai_gym.gym_available_platforms import OpenAIGymAvailablePlatformTypes

class TestSensor(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=config['property_class'])

    def _calculate_measurement(self, state):
        return np.array([1.0, 1.0]).astype(np.float32)

PluginLibrary.AddClassToGroup(
    TestSensor, "Sensor_Test", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


class TestSensor2D(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=config['property_class'])

    def _calculate_measurement(self, state):
        return np.array([[1.0, 1.0], [1.5, 1.5]]).astype(np.float32)

PluginLibrary.AddClassToGroup(
    TestSensor2D, "Sensor_Test2D", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


def build_observe_sensor(sensorname, sensorclass, propconfig, output_units=None):

    sensorconfig = {
        "parent_platform": "none",
        "config": {},
        "property_class": BoxProp,
        "properties": propconfig
    }

    platformconfig = {
        "platform_name": "blue0",
        "platform": "none",
        "parts_list": [(sensorclass, sensorconfig)]
    }

    class TestPlatform(BasePlatform):
        def __init__(self, platform_name, platform, parts_list):
            super().__init__(platform_name=platform_name, platform=platform, parts_list=parts_list)

        @property
        def operable(self):
            return True

    observesensorconfig = {
        "name": "ObserveSensor",
        "sensor": sensorname,
        "normalization": {"enabled": False},
        "agent_name": "blue0",
        "platform": TestPlatform(**platformconfig),
        "output_units": output_units
    }

    return ObserveSensor(**observesensorconfig)

# No output units specified, sensor has default units
def test_observe_sensor_no_unit_default():

    propconfig = {
        "name": "TestProp",
        "low": [1.0, 1.0],
        "high": [2.0, 2.0],
        "unit": ["meter", "radian"],
        "description": "Test Space"
    }

    observe_sensor = build_observe_sensor("Sensor_Test", TestSensor, propconfig)

    # Check observation_units
    observation_units = observe_sensor.observation_units()[observe_sensor.Fields.DIRECT_OBSERVATION]
    assert observation_units == ["m", "rad"]
    # Check observation_space
    observation_space = observe_sensor.observation_space()[observe_sensor.Fields.DIRECT_OBSERVATION]
    np.testing.assert_array_equal(observation_space.low, np.array([1.0, 1.0]).astype(np.float32))
    np.testing.assert_array_equal(observation_space.high, np.array([2.0, 2.0]).astype(np.float32))
    # Check get_observation
    observe_sensor._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor.get_observation({}, {}, {})[observe_sensor.Fields.DIRECT_OBSERVATION]
    np.testing.assert_array_equal(observation, np.array([1.0, 1.0]).astype(np.float32))


# No output units specified, sensor does not have default units
def test_observe_sensor_no_unit_not_default():

    propconfig = {
        "name": "TestProp",
        "low": [1.0, 1.0],
        "high": [2.0, 2.0],
        "unit": ["feet", "degree"],
        "description": "Test Space"
    }

    observe_sensor = build_observe_sensor("Sensor_Test", TestSensor, propconfig)

    # Check observation_units
    observation_units = observe_sensor.observation_units()[observe_sensor.Fields.DIRECT_OBSERVATION]
    assert observation_units == ["m", "rad"]
    # Check observation_space
    observation_space = observe_sensor.observation_space()[observe_sensor.Fields.DIRECT_OBSERVATION]
    np.testing.assert_array_equal(observation_space.low, np.array([1.0 / 3.28084, 1.0 * math.pi / 180.0]).astype(np.float32))
    np.testing.assert_array_equal(observation_space.high, np.array([2.0 / 3.28084, 2.0 * math.pi / 180.0]).astype(np.float32))
    # Check get_observation
    observe_sensor._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor.get_observation({}, {}, {})[observe_sensor.Fields.DIRECT_OBSERVATION]
    np.testing.assert_array_equal(observation, np.array([1.0 / 3.28084, 1.0 * math.pi / 180.0]).astype(np.float32))


# Output units specified, sensor has default units
def test_observe_sensor_unit_default():

    propconfig = {
        "name": "TestProp",
        "low": [1.0, 1.0],
        "high": [2.0, 2.0],
        "unit": ["meter", "radian"],
        "description": "Test Space"
    }

    output_units = ["feet", "degree"]
    observe_sensor = build_observe_sensor("Sensor_Test", TestSensor, propconfig, output_units)

    # Check observation_units
    observation_units = observe_sensor.observation_units()[observe_sensor.Fields.DIRECT_OBSERVATION]
    assert observation_units == ["ft", "deg"]
    # Check observation_space
    observation_space = observe_sensor.observation_space()[observe_sensor.Fields.DIRECT_OBSERVATION]
    np.testing.assert_array_equal(observation_space.low, np.array([3.28084, 180.0 / math.pi]).astype(np.float32))
    np.testing.assert_array_equal(observation_space.high, np.array([2.0 * 3.28084, 2.0 * 180.0 / math.pi]).astype(np.float32))
    # Check get_observation
    observe_sensor._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor.get_observation({}, {}, {})[observe_sensor.Fields.DIRECT_OBSERVATION]
    np.testing.assert_array_equal(observation, np.array([3.28084, 180.0 / math.pi]).astype(np.float32))


# Output units specified, sensor does not have default units
def test_observe_sensor_unit_not_default():

    propconfig = {
        "name": "TestProp",
        "low": [1.0, 1.0],
        "high": [2.0, 2.0],
        "unit": ["feet", "degree"],
        "description": "Test Space"
    }

    output_units = ["nm", "degree"]
    observe_sensor = build_observe_sensor("Sensor_Test", TestSensor, propconfig, output_units)

    # Check observation_units
    observation_units = observe_sensor.observation_units()[observe_sensor.Fields.DIRECT_OBSERVATION]
    assert observation_units == ["nm", "deg"]
    # Check observation_space
    observation_space = observe_sensor.observation_space()[observe_sensor.Fields.DIRECT_OBSERVATION]
    np.testing.assert_array_equal(observation_space.low, np.array([1.0 / 1852 / 3.28084, 1.0]).astype(np.float32))
    np.testing.assert_array_equal(observation_space.high, np.array([2.0 / 1852 / 3.28084, 2.0]).astype(np.float32))
    # Check get_observation
    observe_sensor._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor.get_observation({}, {}, {})[observe_sensor.Fields.DIRECT_OBSERVATION]
    np.testing.assert_array_equal(observation, np.array([1.0 / 1852 / 3.28084, 1.0]).astype(np.float32))


# Output units specified, sensor does not have default units, 2D case
def test_observe_sensor_unit_not_default_2d():

    propconfig = {
        "name": "TestProp",
        "low": [[1.0, 1.0], [1.5, 1.5]],
        "high": [[2.0, 2.0], [2.5, 2.5]],
        "unit": [["feet", "degree"], ["feet", "degree"]],
        "description": "Test Space"
    }

    output_units = [["nm", "rad"], ["nm", "rad"]]
    observe_sensor = build_observe_sensor("Sensor_Test2D", TestSensor2D, propconfig, output_units)

    # Check observation_units
    observation_units = observe_sensor.observation_units()[observe_sensor.Fields.DIRECT_OBSERVATION]
    assert observation_units == [["nm", "rad"], ["nm", "rad"]]
    # Check observation_space
    observation_space = observe_sensor.observation_space()[observe_sensor.Fields.DIRECT_OBSERVATION]
    np.testing.assert_array_equal(observation_space.low, np.array([[1.0 / 1852 / 3.28084, 1.0 * math.pi / 180.0], [1.5 / 1852 / 3.28084, 1.5 * math.pi / 180.0]]).astype(np.float32))
    np.testing.assert_array_equal(observation_space.high, np.array([[2.0 / 1852 / 3.28084, 2.0 * math.pi / 180.0], [2.5 / 1852 / 3.28084, 2.5 * math.pi / 180.0]]).astype(np.float32))
    # Check get_observation
    observe_sensor._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor.get_observation({}, {}, {})[observe_sensor.Fields.DIRECT_OBSERVATION]
    expected_observation = np.array([[1.0 / 1852 / 3.28084, 1.0 * math.pi / 180.0], [1.5 / 1852 / 3.28084, 1.5 * math.pi / 180.0]]).astype(np.float32)
    np.testing.assert_array_equal(observation, expected_observation)


# No output units specified, sensor does not have default units, 2D case
def test_observe_sensor_no_unit_not_default_2d():

    propconfig = {
        "name": "TestProp",
        "low": [[1.0, 1.0], [1.5, 1.5]],
        "high": [[2.0, 2.0], [2.5, 2.5]],
        "unit": [["feet", "degree"], ["feet", "degree"]],
        "description": "Test Space"
    }

    observe_sensor = build_observe_sensor("Sensor_Test2D", TestSensor2D, propconfig)

    # Check observation_units
    observation_units = observe_sensor.observation_units()[observe_sensor.Fields.DIRECT_OBSERVATION]
    assert observation_units == [["m", "rad"], ["m", "rad"]]
    # Check observation_space
    observation_space = observe_sensor.observation_space()[observe_sensor.Fields.DIRECT_OBSERVATION]
    np.testing.assert_array_equal(observation_space.low, np.array([[1.0 / 3.28084, 1.0 * math.pi / 180.0], [1.5 / 3.28084, 1.5 * math.pi / 180.0]]).astype(np.float32))
    np.testing.assert_array_equal(observation_space.high, np.array([[2.0 / 3.28084, 2.0 * math.pi / 180.0], [2.5 / 3.28084, 2.5 * math.pi / 180.0]]).astype(np.float32))
    # Check get_observation
    observe_sensor._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor.get_observation({}, {}, {})[observe_sensor.Fields.DIRECT_OBSERVATION]
    expected_observation = np.array([[1.0 / 3.28084, 1.0 * math.pi / 180.0], [1.5 / 3.28084, 1.5 * math.pi / 180.0]]).astype(np.float32)
    np.testing.assert_array_equal(observation, expected_observation)
