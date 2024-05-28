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

from corl.glues.base_dict_wrapper import BaseDictWrapperGlue
from corl.glues.base_multi_wrapper import BaseMultiWrapperGlue
from corl.glues.base_wrapper import BaseWrapperGlue
from corl.libraries.functor import Functor, FunctorDictWrapper, FunctorWrapper
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp
from corl.libraries.units import Quantity, corl_get_ureg, corl_quantity
from corl.simulators.base_parts import BaseController, BaseSensor
from corl.simulators.gymnasium.gymnasium_available_platforms import GymnasiumAvailablePlatformTypeMain
from corl.simulators.gymnasium.gymnasium_simulator import GymnasiumPlatform, GymnasiumSimulator


class Test1dController(BaseController):
    def __init__(self, parent_platform, config):
        super().__init__(property_class=BoxProp, parent_platform=parent_platform, config=config)

    def apply_control(self, control):
        assert isinstance(control, Quantity)
        self.parent_platform.save_action_to_platform(action=control)

    def get_applied_control(self) -> Quantity:
        return self.parent_platform.get_applied_action()


PluginLibrary.AddClassToGroup(
    Test1dController, "Controller_Test1dController", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


class TestSensor1DArray(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=BoxProp)

    def _calculate_measurement(self, state):  # noqa: PLR6301
        return corl_get_ureg().Quantity(np.array([50.0]).astype(np.float32), "dimensionless")


PluginLibrary.AddClassToGroup(
    TestSensor1DArray, "Sensor_Test1DArray", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


class TestSensor2DArray(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=BoxProp)

    def _calculate_measurement(self, state):  # noqa: PLR6301
        return corl_get_ureg().Quantity(np.array([50.0, 25.0]).astype(np.float32), "dimensionless")


PluginLibrary.AddClassToGroup(
    TestSensor2DArray, "Sensor_Test2DArray", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


class TestSensor3DArray(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=BoxProp)

    def _calculate_measurement(self, state):  # noqa: PLR6301
        return corl_get_ureg().Quantity(np.array([50.0, 25.0, 12.5]).astype(np.float32), "dimensionless")


PluginLibrary.AddClassToGroup(
    TestSensor3DArray, "Sensor_Test3DArray", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


class TestSensorAngle(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__(parent_platform=parent_platform, config=config, property_class=BoxProp)

    def _calculate_measurement(self, state):  # noqa: PLR6301
        return corl_get_ureg().Quantity(np.array([np.pi / 4.0]).astype(np.float32), "rad")


PluginLibrary.AddClassToGroup(
    TestSensorAngle, "TestSensorAngle", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)


@pytest.fixture(name="platform_name")
def get_platform_name():
    return "test_platform"


@pytest.fixture(name="parts_list")
def get_parts_list():
    return [
        (
            Test1dController,
            {
                "properties": {
                    "name": "Test1dController",
                    "description": "Test1dController",
                    "low": [-1.0],
                    "high": [1.0],
                    "unit": "foot / second",
                }
            },
        ),
        (
            TestSensor1DArray,
            {
                "properties": {
                    "name": "TestSensor1DArray",
                    "description": "TestSensor1DArray",
                    "low": [0.0],
                    "high": [100.0],
                    "unit": "dimensionless",
                }
            },
        ),
        (
            TestSensor2DArray,
            {
                "properties": {
                    "name": "TestSensor2DArray",
                    "description": "TestSensor2DArray",
                    "low": [0.0, 0.0],
                    "high": [100.0, 50.0],
                    "unit": "dimensionless",
                }
            },
        ),
        (
            TestSensor3DArray,
            {
                "properties": {
                    "name": "TestSensor3DArray",
                    "description": "TestSensor3DArray",
                    "low": [0.0, 0.0, 0.0],
                    "high": [100.0, 50.0, 25.0],
                    "unit": "dimensionless",
                }
            },
        ),
        (
            TestSensorAngle,
            {"properties": {"name": "TestSensorAngle", "description": " ", "low": [0.0], "high": [2 * np.pi], "unit": "rad"}},
        ),
    ]


@pytest.fixture(name="platform")
def get_platform(platform_name, parts_list):
    return GymnasiumPlatform(platform_name=platform_name, platform=gymnasium.make("MountainCarContinuous-v0"), parts_list=parts_list)


@pytest.fixture(name="platform_config")
def get_platform_config():
    return {"platform_class": "corl.simulators.gymnasium.gymnasium_simulator.GymnasiumInclusivePartsPlatform"}


def observe_sensor(sensor_name: str):
    return {
        "functor": "corl.glues.common.observe_sensor.ObserveSensor",
        "config": {
            "sensor": sensor_name,
        },
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
            },
        },
    }


def trig_values_glue(sensor_name: str):
    return {
        "functor": "corl.glues.common.trig_values.TrigValues",
        "wrapped": {
            "functor": "corl.glues.common.observe_sensor.ObserveSensor",
            "config": {
                "sensor": sensor_name,
            },
        },
    }


def mag_glue(sensor_name: str):
    return {
        "functor": "corl.glues.common.magnitude.MagnitudeGlue",
        "wrapped": {
            "functor": "corl.glues.common.observe_sensor.ObserveSensor",
            "config": {
                "sensor": sensor_name,
            },
        },
    }


def part_validity_glue(sensor_name: str):
    return {
        "functor": "corl.glues.common.observe_part_validity.ObservePartValidity",
        "config": {
            "part": sensor_name,
        },
    }


def projected_quantity_glue():
    return {
        "functor": "corl.glues.common.projected_quantity.ProjectedQuantity",
        "config": {},
        "wrapped": {
            "quantity": {
                "functor": "corl.glues.common.target_value.TargetValue",
                "config": {
                    "target_value": {"value": 1.0, "unit": "foot / second"},
                    "limit": {
                        "minimum": {
                            "value": -10.0,
                            "unit": "foot / second",
                        },
                        "maximum": {
                            "value": 10.0,
                            "unit": "foot / second",
                        },
                    },
                },
            },
            "angle1": {
                "functor": "corl.glues.common.target_value.TargetValue",
                "config": {
                    "target_value": {"value": 0.5, "unit": "rad"},
                    "limit": {
                        "minimum": {
                            "value": -3.1416,
                            "unit": "rad",
                        },
                        "maximum": {
                            "value": 3.1416,
                            "unit": "rad",
                        },
                    },
                },
            },
        },
    }


def target_value_glue():
    return {
        "functor": "corl.glues.common.target_value.TargetValue",
        "config": {
            "target_value": {"value": 1.0, "unit": "foot / second"},
            "limit": {
                "minimum": {
                    "value": -10.0,
                    "unit": "foot / second",
                },
                "maximum": {
                    "value": 10.0,
                    "unit": "foot / second",
                },
            },
        },
    }


def controller_glue(controller_name):
    return {
        "functor": "corl.glues.common.controller_glue.ControllerGlue",
        "config": {"controller": controller_name},
    }


@pytest.mark.parametrize(
    "config, normalized_obs_space, raw_observation, functor_class",
    [
        pytest.param(
            observe_sensor(sensor_name="Sensor_Test1DArray"),
            gymnasium.spaces.Dict({"direct_observation": gymnasium.spaces.Box(low=0.0, high=100.0, shape=(1,))}),
            OrderedDict({"direct_observation": corl_quantity()(np.array([50.0], dtype=np.float32), "dimensionless")}),
            Functor,
            id="1d_observe_sensor",
        ),
        pytest.param(
            unit_vec_standardnorm(sensor_name="Sensor_Test1DArray", array_size=1),
            gymnasium.spaces.Dict(
                {
                    "unit_vec": gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
                }
            ),
            OrderedDict({"unit_vec": corl_quantity()(np.array([1.0], dtype=np.float32), "dimensionless")}),
            FunctorWrapper,
            id="1d_unit_vector_std_norm",
        ),
        pytest.param(
            unit_vec_standardnorm(sensor_name="Sensor_Test2DArray", array_size=2),
            gymnasium.spaces.Dict(
                {
                    "unit_vec": gymnasium.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])),
                }
            ),
            OrderedDict({"unit_vec": corl_quantity()(np.array([0.8944272, 0.4472136], dtype=np.float32), "dimensionless")}),
            FunctorWrapper,
            id="2d_unit_vector_std_norm",
        ),
        pytest.param(
            unit_vec_standardnorm(sensor_name="Sensor_Test3DArray", array_size=3),
            gymnasium.spaces.Dict(
                {
                    "unit_vec": gymnasium.spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0])),
                }
            ),
            OrderedDict({"unit_vec": corl_quantity()(np.array([0.8728716, 0.4364358, 0.2182179], dtype=np.float32), "dimensionless")}),
            FunctorWrapper,
            id="3d_unit_vector_std_norm",
        ),
        pytest.param(
            trig_values_glue(sensor_name="TestSensorAngle"),
            gymnasium.spaces.Dict(
                {
                    "cos": gymnasium.spaces.Box(low=np.array([-1.0]), high=np.array([1.0])),
                    "sin": gymnasium.spaces.Box(low=np.array([-1.0]), high=np.array([1.0])),
                }
            ),
            OrderedDict(
                {
                    "cos": corl_quantity()(np.array([np.cos(np.pi / 4)], dtype=np.float32), "dimensionless"),
                    "sin": corl_quantity()(np.array([np.cos(np.pi / 4)], dtype=np.float32), "dimensionless"),
                }
            ),
            FunctorWrapper,
            id="trig_values_linear",
        ),
    ],
)
def test_obs_sensor_glue(config, normalized_obs_space, raw_observation, platform, functor_class):
    functor = functor_class(**config)
    created_glue = functor.create_functor_object(platform=platform, agent_name="test_agent")

    if isinstance(created_glue, BaseWrapperGlue):
        created_glue.glue()._sensor.calculate_and_cache_measurement({})  # noqa: SLF001
    elif isinstance(created_glue, BaseMultiWrapperGlue):
        for glue in created_glue.glues():
            glue._sensor.calculate_and_cache_measurement({})  # noqa: SLF001
    elif isinstance(created_glue, BaseDictWrapperGlue):
        for _name, glue in created_glue.glues():
            glue._sensor.calculate_and_cache_measurement({})  # noqa: SLF001
    else:
        created_glue._sensor.calculate_and_cache_measurement({})  # noqa: SLF001

    check_observation_glue(created_glue, normalized_obs_space, raw_observation)


@pytest.mark.parametrize(
    "config, normalized_obs_space, raw_observation, functor_class",
    [
        pytest.param(
            part_validity_glue(sensor_name="Sensor_Test2DArray"),
            gymnasium.spaces.Dict(
                {
                    "validity_observation": gymnasium.spaces.Discrete(n=2),
                }
            ),
            OrderedDict(
                {
                    "validity_observation": corl_quantity()(1, "dimensionless"),
                }
            ),
            Functor,
            id="part_validity",
        ),
        pytest.param(
            projected_quantity_glue(),
            gymnasium.spaces.Dict(
                {
                    "projected_quantity": gymnasium.spaces.Box(low=np.array([-10.0]), high=np.array([10])),
                }
            ),
            OrderedDict({"projected_quantity": corl_quantity()(np.array([0.87758255], dtype=np.float32), "foot / second")}),
            FunctorDictWrapper,
            id="projected_quantity",
        ),
        pytest.param(
            target_value_glue(),
            gymnasium.spaces.Dict(
                {
                    "target_value": gymnasium.spaces.Box(low=np.array([-10.0]), high=np.array([10])),
                }
            ),
            OrderedDict({"target_value": corl_quantity()(np.array([1], dtype=np.float32), "foot / second")}),
            Functor,
            id="target_values",
        ),
    ],
)
def test_glue(config, normalized_obs_space, raw_observation, platform, functor_class):
    functor = functor_class(**config)
    created_glue = functor.create_functor_object(platform=platform, agent_name="test_agent")
    check_observation_glue(created_glue, normalized_obs_space, raw_observation)


@pytest.mark.parametrize(
    "config, action_space, action, observation_space, observation, functor_class",
    [
        pytest.param(
            controller_glue(controller_name="Controller_Test1dController"),
            gymnasium.spaces.Box(low=np.array([-1.0]), high=np.array([1.0])),
            corl_quantity()(np.array([1], dtype=np.float32), "foot / second"),
            gymnasium.spaces.Dict({"control": gymnasium.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]))}),
            OrderedDict({"control": corl_quantity()(np.array([1], dtype=np.float32), "foot / second")}),
            Functor,
            id="controller_glue",
        ),
    ],
)
def test_contoller_glue(config, action_space, action, observation_space, observation, platform, functor_class):
    functor = functor_class(**config)
    created_glue = functor.create_functor_object(platform=platform, agent_name="test_agent")

    assert action_space == created_glue.action_space
    assert observation_space == created_glue.observation_space
    assert created_glue.action_prop.contains(action)
    assert created_glue.observation_prop.contains(observation)

    created_glue.apply_action(action, {}, {}, {}, {})
    applied_action = created_glue.get_applied_control()
    observation = created_glue.get_observation({}, {}, {})

    assert action == applied_action
    assert action == observation["control"]
