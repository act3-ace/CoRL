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

import gymnasium
import numpy as np
import pytest
from glue_test_utils import check_observation_glue
from tree import map_structure

from corl.glues.common.observe_sensor import ObserveSensor
from corl.libraries.functor import FunctorDictWrapper
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp
from corl.libraries.units import corl_quantity
from corl.simulators.base_parts import BaseSensor
from corl.simulators.base_platform import BasePlatform
from corl.simulators.gymnasium.gymnasium_available_platforms import GymnasiumAvailablePlatformTypeMain
from corl.simulators.gymnasium.gymnasium_simulator import GymnasiumSimulator


class TestSensor(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=config["property_class"])

    def _calculate_measurement(self, state):
        return corl_quantity()(np.array([1.0, 1.0]).astype(np.float32), self.measurement_properties.get_units())


PluginLibrary.AddClassToGroup(
    TestSensor, "Sensor_Test", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


class TestAngleSensor(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=config["property_class"])

    def _calculate_measurement(self, state):
        return corl_quantity()(np.array([1080.0]).astype(np.float32), self.measurement_properties.get_units())


PluginLibrary.AddClassToGroup(
    TestAngleSensor, "AngleSensor_Test", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


def build_observe_sensor(sensorname, sensorclass, propconfig, target_value_diff_config, output_units=None):
    sensorconfig = {"parent_platform": "none", "config": {}, "property_class": BoxProp, "properties": propconfig}

    platformconfig = {"platform_name": "blue0", "platform": "none", "parts_list": [(sensorclass, sensorconfig)]}

    class TestPlatform(BasePlatform):
        def __init__(self, platform_name, platform, parts_list):
            super().__init__(platform_name=platform_name, platform=platform, parts_list=parts_list)

        @property
        def operable(self):
            return True

    platform = TestPlatform(**platformconfig)
    functor = FunctorDictWrapper(**target_value_diff_config)
    return functor.create_functor_object(platform=platform, agent_name="test_agent")


propconfig = {"name": "TestProp", "low": [1.0, 1.0], "high": [2.0, 2.0], "unit": "meter", "description": "Test Space"}

target_value_diff_config_distance = {
    "functor": "corl.glues.common.target_value_difference.TargetValueDifference",
    "wrapped": {"sensor": {"functor": "corl.glues.common.observe_sensor.ObserveSensor", "config": {"sensor": "Sensor_Test"}}},
    "config": {
        "target_value": {"value": [1.6, 1.6], "unit": "meter"},
        "index": 0,
        "limit": {
            "minimum": {
                "value": [-3.0, -3.0],
                "unit": "meter",
            },
            "maximum": {
                "value": [3.0, 3.0],
                "unit": "meter",
            },
            "unit": "meter",
        },
    },
}

propconfig_angle = {"name": "TestProp", "low": [0.0], "high": [350], "unit": "degree", "description": "Test Space"}


target_value_diff_config_angle = {
    "functor": "corl.glues.common.target_value_difference.TargetValueDifference",
    "wrapped": {"sensor": {"functor": "corl.glues.common.observe_sensor.ObserveSensor", "config": {"sensor": "AngleSensor_Test"}}},
    "config": {
        "is_wrap": True,
        "target_value": {"value": [20.0], "unit": "degree"},
        "index": 0,
        "limit": {
            "minimum": {
                "value": [-180.0],
                "unit": "degree",
            },
            "maximum": {
                "value": [180.0],
                "unit": "degree",
            },
            "unit": "degree",
        },
    },
}


@pytest.mark.parametrize(
    "sensor_name, sensor_class, propconfig, sensor_config, expected_space, expected_obs, expected_units",
    [
        pytest.param(
            "Sensor_Test",
            TestSensor,
            propconfig,
            target_value_diff_config_distance,
            gymnasium.spaces.Dict(
                {"direct_observation_diff": gymnasium.spaces.Box(low=np.array([-3.0, -3.0]), high=np.array([3.0, 3.0]), dtype=np.float32)}
            ),
            OrderedDict({"direct_observation_diff": np.array([0.6, 0.6], dtype=np.float32)}),
            OrderedDict({"direct_observation_diff": "meter"}),
            id="distance_diff",
        ),
        pytest.param(
            "AngleSensor_Test",
            TestAngleSensor,
            propconfig_angle,
            target_value_diff_config_angle,
            gymnasium.spaces.Dict(
                {"direct_observation_diff": gymnasium.spaces.Box(low=np.array([-180.0]), high=np.array([180.0]), dtype=np.float32)}
            ),
            OrderedDict({"direct_observation_diff": np.array([20.0], dtype=np.float32)}),
            OrderedDict({"direct_observation_diff": "degree"}),
            id="angle_diff",
        ),
    ],
)
def test_observe_sensor(sensor_name, sensor_class, propconfig, sensor_config, expected_space, expected_obs, expected_units):
    observe_sensor = build_observe_sensor(sensor_name, sensor_class, propconfig, sensor_config)
    new_expected_obs = map_structure(lambda x, y: corl_quantity()(x, y), expected_obs, expected_units)
    for glue in observe_sensor.glues().values():
        if isinstance(glue, ObserveSensor):
            glue._sensor.calculate_and_cache_measurement(None)  # noqa: SLF001
    check_observation_glue(observe_sensor, expected_space, new_expected_obs)
