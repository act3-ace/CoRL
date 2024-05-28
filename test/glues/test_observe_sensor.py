# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

from collections import OrderedDict

import gymnasium
import numpy as np
import pytest
from glue_test_utils import check_observation_glue

from corl.glues.common.observe_sensor import ObserveSensor
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


class TestSensor2D(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=config["property_class"])

    def _calculate_measurement(self, state):
        return corl_quantity()(np.array([[1.0, 1.0], [1.5, 1.5]]).astype(np.float32), self.measurement_properties.get_units())


PluginLibrary.AddClassToGroup(
    TestSensor2D, "Sensor_Test2D", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


def build_observe_sensor(sensorname, sensorclass, propconfig, output_units=None):
    sensorconfig = {"parent_platform": "none", "config": {}, "property_class": BoxProp, "properties": propconfig}

    platformconfig = {"platform_name": "blue0", "platform": "none", "parts_list": [(sensorclass, sensorconfig)]}

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
        "output_units": output_units,
    }

    return ObserveSensor(**observesensorconfig)


propconfig_no_unit_default = {"name": "TestProp", "low": [1.0, 1.0], "high": [2.0, 2.0], "unit": "meter", "description": "Test Space"}

propconfig_unit_not_default = {"name": "TestProp", "low": [1.0, 0.0], "high": [2.0, 10.0], "unit": "foot", "description": "Test Space"}

propconfig_not_default_2d = {
    "name": "TestProp",
    "low": [[1.0, 0.0], [1.5, 0.5]],
    "high": [[2.0, 10.0], [2.5, 6.5]],
    "unit": "foot",
    "description": "Test Space",
}

propconfig_no_unit_not_default_2d = {
    "name": "TestProp",
    "low": [[1.0, 1.0], [1.5, 1.5]],
    "high": [[2.0, 2.0], [2.5, 2.5]],
    "unit": "foot",
    "description": "Test Space",
}


@pytest.mark.parametrize(
    "observe_sensor, expected_space, expected_obs",
    [
        pytest.param(
            build_observe_sensor("Sensor_Test", TestSensor, propconfig_no_unit_default),
            gymnasium.spaces.Dict(
                {"direct_observation": gymnasium.spaces.Box(low=np.array([1.0, 1.0]), high=np.array([2.0, 2.0]), dtype=np.float32)}
            ),
            OrderedDict({"direct_observation": corl_quantity()(np.array([1.0, 1.0], dtype=np.float32), "meter")}),
            id="no_unit_default",
        ),
        pytest.param(
            build_observe_sensor("Sensor_Test", TestSensor, propconfig_unit_not_default, "nmi"),
            gymnasium.spaces.Dict(
                {
                    "direct_observation": gymnasium.spaces.Box(
                        low=np.array([1.0 / 1852 / 3.28084, 0.0]),
                        high=np.array([2.0 / 1852 / 3.28084, 10.0 / 1852 / 3.28084]),
                        dtype=np.float32,
                    )
                }
            ),
            OrderedDict(
                {"direct_observation": corl_quantity()(np.array([1.0 / 1852 / 3.28084, 1.0 / 1852 / 3.28084], dtype=np.float32), "nmi")}
            ),
            id="unit_not_default",
        ),
        pytest.param(
            build_observe_sensor("Sensor_Test2D", TestSensor2D, propconfig_not_default_2d, "nmi"),
            gymnasium.spaces.Dict(
                {
                    "direct_observation": gymnasium.spaces.Box(
                        low=np.array([[1.0 / 1852 / 3.28084, 0.0 / 1852 / 3.28084], [1.5 / 1852 / 3.28084, 0.5 / 1852 / 3.28084]]),
                        high=np.array([[2.0 / 1852 / 3.28084, 10.0 / 1852 / 3.28084], [2.5 / 1852 / 3.28084, 6.5 / 1852 / 3.28084]]),
                        dtype=np.float32,
                    )
                }
            ),
            OrderedDict(
                {
                    "direct_observation": corl_quantity()(
                        np.array(
                            [[1.0 / 1852 / 3.28084, 1.0 / 1852 / 3.28084], [1.5 / 1852 / 3.28084, 1.5 / 1852 / 3.28084]], dtype=np.float32
                        ),
                        "nmi",
                    )
                }
            ),
            id="not_default_2d",
        ),
        pytest.param(
            build_observe_sensor("Sensor_Test2D", TestSensor2D, propconfig_no_unit_not_default_2d),
            gymnasium.spaces.Dict(
                {
                    "direct_observation": gymnasium.spaces.Box(
                        low=np.array([[1.0, 1.0], [1.5, 1.5]]), high=np.array([[2.0, 2.0], [2.5, 2.5]]), dtype=np.float32
                    )
                }
            ),
            OrderedDict({"direct_observation": corl_quantity()(np.array([[1.0, 1.0], [1.5, 1.5]], dtype=np.float32), "foot")}),
            id="no_unit_not_default_2d",
        ),
    ],
)
def test_observe_sensor(observe_sensor, expected_space, expected_obs):
    observe_sensor._sensor.calculate_and_cache_measurement(None)  # noqa: SLF001
    check_observation_glue(observe_sensor, expected_space, expected_obs)
