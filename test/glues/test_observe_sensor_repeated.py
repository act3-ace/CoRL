"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import math
from collections import OrderedDict

import gymnasium
import numpy as np
import pytest
from glue_test_utils import check_observation_glue
from pydantic import BaseModel, ImportString, model_validator
from ray.rllib.utils.spaces.repeated import Repeated

from corl.glues.common.observe_sensor_repeated import ObserveSensorRepeated
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp, Prop, RepeatedProp
from corl.libraries.units import corl_quantity
from corl.simulators.base_parts import BaseSensor
from corl.simulators.base_platform import BasePlatform
from corl.simulators.gymnasium.gymnasium_available_platforms import GymnasiumAvailablePlatformTypeMain
from corl.simulators.gymnasium.gymnasium_simulator import GymnasiumSimulator


class TrackElement(BaseModel):
    functor: ImportString
    config: dict = {}


class RepeatedSensorProp(RepeatedProp):
    name: str = "test_repeated_prop"
    max_len: int = 100
    track_elements: list[TrackElement] = []
    description: str = "test repeated prop"
    child_space: dict[str, Prop] = {}

    @model_validator(mode="after")
    def generate_child_space(self):
        track_elements = self.track_elements
        child_space = dict()

        for track_element in track_elements:
            prop = track_element.functor(**track_element.config)
            child_space[prop.name] = prop
        self.child_space = child_space
        return self


class TestSensorRepeated(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=config["property_class"])

    def _calculate_measurement(self, state):
        ret_data: list[dict] = []
        idx = 1
        plat_dict: dict = OrderedDict()

        for obs_property in self._properties.child_space.values():
            plat_dict[obs_property.name] = corl_quantity()(np.array([idx]).astype(np.float32), obs_property.get_units())
            idx += 1
        ret_data.append(plat_dict)
        return ret_data


PluginLibrary.AddClassToGroup(
    TestSensorRepeated, "Sensor_Test_Repeated", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


def build_observe_sensor_repeated(sensorname, sensorclass, propconfig, output_units=None):
    if output_units is None:
        output_units = {}
    sensorconfig = {"parent_platform": "none", "config": {}, "property_class": RepeatedSensorProp, "properties": propconfig}

    platformconfig = {"platform_name": "blue0", "platform": "none", "parts_list": [(sensorclass, sensorconfig)]}

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


propconfig1 = {
    "track_elements": [
        {"functor": BoxProp, "config": {"name": "TestProp1", "low": [1.0], "high": [2.0], "unit": "meter", "description": "Test Space 1"}},
        {"functor": BoxProp, "config": {"name": "TestProp2", "low": [1.0], "high": [2.0], "unit": "radian", "description": "Test Space 2"}},
    ]
}

propconfig2 = {
    "track_elements": [
        {"functor": BoxProp, "config": {"name": "TestProp1", "low": [1.0], "high": [2.0], "unit": "foot", "description": "Test Space 1"}},
        {"functor": BoxProp, "config": {"name": "TestProp2", "low": [1.0], "high": [2.0], "unit": "degree", "description": "Test Space 2"}},
    ]
}


@pytest.mark.parametrize(
    "observe_sensor_repeated, expected_space, expected_obs",
    [
        pytest.param(
            build_observe_sensor_repeated("Sensor_Test_Repeated", TestSensorRepeated, propconfig1),
            gymnasium.spaces.Dict(
                {
                    "direct_observation": Repeated(
                        child_space=gymnasium.spaces.Dict(
                            {
                                "TestProp1": gymnasium.spaces.Box(low=np.array([1.0]), high=np.array([2.0]), dtype=np.float32),
                                "TestProp2": gymnasium.spaces.Box(low=np.array([1.0]), high=np.array([2.0]), dtype=np.float32),
                            }
                        ),
                        max_len=10,
                    )
                }
            ),
            OrderedDict(
                {
                    "direct_observation": [
                        {
                            "TestProp1": corl_quantity()(np.array([1.0], dtype=np.float32), "meter"),
                            "TestProp2": corl_quantity()(np.array([2.0], dtype=np.float32), "rad"),
                        }
                    ]
                }
            ),
            id="no_unit_default",
        ),
        pytest.param(
            build_observe_sensor_repeated(
                "Sensor_Test_Repeated", TestSensorRepeated, propconfig1, {"TestProp1": "foot", "TestProp2": "degree"}
            ),
            gymnasium.spaces.Dict(
                {
                    "direct_observation": Repeated(
                        child_space=gymnasium.spaces.Dict(
                            {
                                "TestProp1": gymnasium.spaces.Box(
                                    low=np.array([1.0 * 3.28084]), high=np.array([2.0 * 3.28084]), dtype=np.float32
                                ),
                                "TestProp2": gymnasium.spaces.Box(
                                    low=np.array([1.0 * 180.0 / math.pi]), high=np.array([2.0 * 180.0 / math.pi]), dtype=np.float32
                                ),
                            }
                        ),
                        max_len=10,
                    )
                }
            ),
            OrderedDict(
                {
                    "direct_observation": [
                        {
                            "TestProp1": corl_quantity()(np.array([1.0 * 3.28084], dtype=np.float32), "foot"),
                            "TestProp2": corl_quantity()(np.array([2.0 / math.pi * 180.0], dtype=np.float32), "degree"),
                        }
                    ]
                }
            ),
            id="unit_default",
        ),
        pytest.param(
            build_observe_sensor_repeated("Sensor_Test_Repeated", TestSensorRepeated, propconfig2, {"TestProp1": "nmi"}),
            gymnasium.spaces.Dict(
                {
                    "direct_observation": Repeated(
                        child_space=gymnasium.spaces.Dict(
                            {
                                "TestProp1": gymnasium.spaces.Box(
                                    low=np.array([1.0 / 1852 / 3.28084]), high=np.array([2.0 / 1852 / 3.28084]), dtype=np.float32
                                ),
                                "TestProp2": gymnasium.spaces.Box(low=np.array([1.0]), high=np.array([2.0]), dtype=np.float32),
                            }
                        ),
                        max_len=10,
                    )
                }
            ),
            OrderedDict(
                {
                    "direct_observation": [
                        {
                            "TestProp1": corl_quantity()(np.array([1.0 / 1852 / 3.28084], dtype=np.float32), "nmi"),
                            "TestProp2": corl_quantity()(np.array([2.0], dtype=np.float32), "degree"),
                        }
                    ]
                }
            ),
            id="partial_unit_default",
        ),
    ],
)
def test_observe_sensor_repeated(observe_sensor_repeated, expected_space, expected_obs):
    observe_sensor_repeated._sensor.calculate_and_cache_measurement(None)  # noqa: SLF001
    check_observation_glue(observe_sensor_repeated, expected_space, expected_obs)
