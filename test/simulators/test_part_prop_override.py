import gymnasium

from corl.simulators.gymnasium.gymnasium_controllers import GymnasiumMainController
from corl.simulators.gymnasium.gymnasium_sensors import GymnasiumStateSensor
from corl.simulators.gymnasium.gymnasium_simulator import GymnasiumPlatform


def test_override_controller_sensor_prop():
    NUM_DISCRETE = 1
    base_environment = gymnasium.make("CartPole-v1")

    # this is a discrete
    base_action_space = base_environment.action_space
    base_obs_space = base_environment.observation_space

    controller_mod_part = (GymnasiumMainController, {"properties": {"n": NUM_DISCRETE}})
    sensor_mod_part = (GymnasiumStateSensor, {"properties": {"high": list(base_obs_space.high / 2), "low": list(base_obs_space.low / 2)}})
    platform_mod_parts = [controller_mod_part, sensor_mod_part]

    openai_mod_platform = GymnasiumPlatform(platform_name="blue0", platform=base_environment, parts_list=platform_mod_parts)

    controller_space = next(iter(openai_mod_platform.controllers.values())).control_properties.create_space()
    sensor_space = next(iter(openai_mod_platform.sensors.values())).measurement_properties.create_space()

    assert controller_space != base_action_space
    assert controller_space == gymnasium.spaces.Discrete(NUM_DISCRETE)
    assert sensor_space != base_obs_space
