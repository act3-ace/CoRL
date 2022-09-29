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
from pydantic import ValidationError, BaseModel, PyObject, root_validator
import pytest
import typing
import math
from collections import OrderedDict

from corl.glues.common.observe_sensor_repeated import ObserveSensorRepeated
from corl.libraries.property import Prop, BoxProp, RepeatedProp
from corl.simulators.base_parts import BaseSensor
from corl.simulators.base_platform import BasePlatform
from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.openai_gym.gym_simulator import OpenAIGymSimulator
from corl.simulators.openai_gym.gym_available_platforms import OpenAIGymAvailablePlatformTypes

class TrackElement(BaseModel):
    functor: PyObject
    config: typing.Dict = {}


class RepeatedSensorProp(RepeatedProp):

    name: str = "test_repeated_prop"
    max_len: int = 100
    track_elements: typing.List[TrackElement] = []
    unit: typing.Union[typing.Sequence[str], typing.Sequence[typing.Sequence[str]]] = ["none"]
    description = "test repeated prop"
    child_space: typing.Dict[str, Prop] = {}

    @root_validator
    def generate_child_space(cls, values):  # pylint: disable=no-self-argument,no-self-use
        track_elements = values['track_elements']
        child_space = dict()

        for track_element in track_elements:
            prop = track_element.functor(**track_element.config)
            child_space[prop.name] = prop
        values['child_space'] = child_space
        return values


class TestSensorRepeated(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=config['property_class'])

    def _calculate_measurement(self, state):
        ret_data: typing.List[typing.Dict] = []
        idx = 1
        plat_dict: typing.Dict = OrderedDict()
        for name, obs_property in self._properties.child_space.items():
            plat_dict[obs_property.name] = np.array([idx]).astype(np.float32)
            idx += 1
        ret_data.append(plat_dict)
        return ret_data

PluginLibrary.AddClassToGroup(
    TestSensorRepeated, "Sensor_Test_Repeated", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


def build_observe_sensor_repeated(sensorname, sensorclass, propconfig, output_units=None):

    sensorconfig = {
        "parent_platform": "none",
        "config": {},
        "property_class": RepeatedSensorProp,
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

    observesensorrepeatedconfig = {
        "name": "ObserveSensorRepeated",
        "agent_name": "blue0",
        "platform": TestPlatform(**platformconfig),
        "normalization": {"enabled": False},
        "sensor": sensorname,
        "output_units": output_units,
        "maxlen": 5,
    }

    return ObserveSensorRepeated(**observesensorrepeatedconfig)

# No output units specified, sensor has default units
def test_observe_sensor_repeated_no_unit_default():

    propconfig = {
        'track_elements': [
            {
                "functor": BoxProp,
                "config": {
                    "name": "TestProp1",
                    "low": [1.0],
                    "high": [2.0],
                    "unit": ["meter"],
                    "description": "Test Space 1"
                }
            },
            {
                "functor": BoxProp,
                "config": {
                    "name": "TestProp2",
                    "low": [1.0],
                    "high": [2.0],
                    "unit": ["radian"],
                    "description": "Test Space 2"
                }
            },
        ]
    }

    observe_sensor_repeated = build_observe_sensor_repeated("Sensor_Test_Repeated", TestSensorRepeated, propconfig)

    # Check observation_units
    observation_units = observe_sensor_repeated.observation_units()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    assert observation_units["TestProp1"] == "m"
    assert observation_units["TestProp2"] == "rad"
    # Check observation_space
    observation_space = observe_sensor_repeated.observation_space()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    np.testing.assert_allclose(observation_space.child_space["TestProp1"].low, np.array([1.0]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp1"].high, np.array([2.0]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp2"].low, np.array([1.0]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp2"].high, np.array([2.0]).astype(np.float32))
    # Check get_observation
    observe_sensor_repeated._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor_repeated.get_observation()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    np.testing.assert_allclose(observation[0]["TestProp1"], np.array([1.0]).astype(np.float32))
    np.testing.assert_allclose(observation[0]["TestProp2"], np.array([2.0]).astype(np.float32))


# No output units specified, sensor does not have default units
def test_observe_sensor_repeated_no_unit_not_default():

    propconfig = {
        'track_elements': [
            {
                "functor": BoxProp,
                "config": {
                    "name": "TestProp1",
                    "low": [1.0],
                    "high": [2.0],
                    "unit": ["feet"],
                    "description": "Test Space 1"
                }
            },
            {
                "functor": BoxProp,
                "config": {
                    "name": "TestProp2",
                    "low": [1.0],
                    "high": [2.0],
                    "unit": ["degree"],
                    "description": "Test Space 2"
                }
            },
        ]
    }

    observe_sensor_repeated = build_observe_sensor_repeated("Sensor_Test_Repeated", TestSensorRepeated, propconfig)

    # Check observation_units
    observation_units = observe_sensor_repeated.observation_units()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    assert observation_units["TestProp1"] == "m"
    assert observation_units["TestProp2"] == "rad"
    # Check observation_space
    observation_space = observe_sensor_repeated.observation_space()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    np.testing.assert_allclose(observation_space.child_space["TestProp1"].low, np.array([1.0 / 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp1"].high, np.array([2.0 / 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp2"].low, np.array([1.0 * math.pi / 180.0]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp2"].high, np.array([2.0 * math.pi / 180.0]).astype(np.float32))
    # Check get_observation
    observe_sensor_repeated._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor_repeated.get_observation()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    np.testing.assert_allclose(observation[0]["TestProp1"], np.array([1.0 / 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation[0]["TestProp2"], np.array([2.0 * math.pi / 180.0]).astype(np.float32))


# Output units specified, sensor has default units
def test_observe_sensor_repeated_unit_default():

    propconfig = {
        'track_elements': [
            {
                "functor": BoxProp,
                "config": {
                    "name": "TestProp1",
                    "low": [1.0],
                    "high": [2.0],
                    "unit": ["meter"],
                    "description": "Test Space 1"
                }
            },
            {
                "functor": BoxProp,
                "config": {
                    "name": "TestProp2",
                    "low": [1.0],
                    "high": [2.0],
                    "unit": ["radian"],
                    "description": "Test Space 2"
                }
            },
        ]
    }

    output_units = {"TestProp1": "feet", "TestProp2": "degree"}
    observe_sensor_repeated = build_observe_sensor_repeated("Sensor_Test_Repeated", TestSensorRepeated, propconfig, output_units)

    # Check observation_units
    observation_units = observe_sensor_repeated.observation_units()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    assert observation_units["TestProp1"] == "ft"
    assert observation_units["TestProp2"] == "deg"
    # Check observation_space
    observation_space = observe_sensor_repeated.observation_space()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    np.testing.assert_allclose(observation_space.child_space["TestProp1"].low, np.array([1.0 * 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp1"].high, np.array([2.0 * 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp2"].low, np.array([1.0 * 180.0 / math.pi]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp2"].high, np.array([2.0 * 180.0 / math.pi]).astype(np.float32))
    # Check get_observation
    observe_sensor_repeated._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor_repeated.get_observation()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    np.testing.assert_allclose(observation[0]["TestProp1"], np.array([1.0 * 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation[0]["TestProp2"], np.array([2.0 / math.pi * 180.0]).astype(np.float32))


# Output units specified, sensor does not have default units
def test_observe_sensor_repeated_unit_not_default():

    propconfig = {
        'track_elements': [
            {
                "functor": BoxProp,
                "config": {
                    "name": "TestProp1",
                    "low": [1.0],
                    "high": [2.0],
                    "unit": ["feet"],
                    "description": "Test Space 1"
                }
            },
            {
                "functor": BoxProp,
                "config": {
                    "name": "TestProp2",
                    "low": [1.0],
                    "high": [2.0],
                    "unit": ["degree"],
                    "description": "Test Space 2"
                }
            },
        ]
    }

    output_units = {"TestProp1": "nm", "TestProp2": "degree"}
    observe_sensor_repeated = build_observe_sensor_repeated("Sensor_Test_Repeated", TestSensorRepeated, propconfig, output_units)

    # Check observation_units
    observation_units = observe_sensor_repeated.observation_units()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    assert observation_units["TestProp1"] == "nm"
    assert observation_units["TestProp2"] == "deg"
    # Check observation_space
    observation_space = observe_sensor_repeated.observation_space()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    np.testing.assert_allclose(observation_space.child_space["TestProp1"].low, np.array([1.0 / 1852 / 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp1"].high, np.array([2.0 / 1852 / 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp2"].low, np.array([1.0]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp2"].high, np.array([2.0]).astype(np.float32))
    # Check get_observation
    observe_sensor_repeated._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor_repeated.get_observation()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    np.testing.assert_allclose(observation[0]["TestProp1"], np.array([1.0 / 1852 / 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation[0]["TestProp2"], np.array([2.0]).astype(np.float32))


# Partial output units specified, sensor has default units
def test_observe_sensor_repeated_partial_unit_default():

    propconfig = {
        'track_elements': [
            {
                "functor": BoxProp,
                "config": {
                    "name": "TestProp1",
                    "low": [1.0],
                    "high": [2.0],
                    "unit": ["meter"],
                    "description": "Test Space 1"
                }
            },
            {
                "functor": BoxProp,
                "config": {
                    "name": "TestProp2",
                    "low": [1.0],
                    "high": [2.0],
                    "unit": ["radian"],
                    "description": "Test Space 2"
                }
            },
        ]
    }

    output_units = {"TestProp1": "feet"}
    observe_sensor_repeated = build_observe_sensor_repeated("Sensor_Test_Repeated", TestSensorRepeated, propconfig, output_units)

    # Check observation_units
    observation_units = observe_sensor_repeated.observation_units()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    assert observation_units["TestProp1"] == "ft"
    assert observation_units["TestProp2"] == "rad"
    # Check observation_space
    observation_space = observe_sensor_repeated.observation_space()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    np.testing.assert_allclose(observation_space.child_space["TestProp1"].low, np.array([1.0 * 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp1"].high, np.array([2.0 * 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp2"].low, np.array([1.0]).astype(np.float32))
    np.testing.assert_allclose(observation_space.child_space["TestProp2"].high, np.array([2.0]).astype(np.float32))
    # Check get_observation
    observe_sensor_repeated._sensor.calculate_and_cache_measurement(None)
    observation = observe_sensor_repeated.get_observation()[observe_sensor_repeated.Fields.DIRECT_OBSERVATION]
    np.testing.assert_allclose(observation[0]["TestProp1"], np.array([1.0 * 3.28084]).astype(np.float32))
    np.testing.assert_allclose(observation[0]["TestProp2"], np.array([2.0]).astype(np.float32))
