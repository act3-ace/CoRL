import gymnasium

from corl.simulators.gymnasium.gymnasium_controllers import GymnasiumMainController
from corl.simulators.gymnasium.gymnasium_sensors import GymnasiumStateSensor
from corl.simulators.gymnasium.gymnasium_simulator import GymnasiumPlatform


def test_openai_controller_prop():
    base_environment = gymnasium.make("CartPole-v1")

    # this is a discrete
    base_action_space = base_environment.action_space
    base_obs_space = base_environment.observation_space

    controller_part = (GymnasiumMainController, {})
    sensor_part = (GymnasiumStateSensor, {})
    base_platform_parts = [controller_part, sensor_part]

    openai_platform = GymnasiumPlatform(platform_name="blue0", platform=base_environment, parts_list=base_platform_parts)

    assert next(iter(openai_platform.controllers.values())).control_properties.create_space() == base_action_space
    assert next(iter(openai_platform.sensors.values())).measurement_properties.create_space() == base_obs_space
