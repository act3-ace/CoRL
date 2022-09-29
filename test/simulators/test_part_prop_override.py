import gym
from corl.simulators.openai_gym.gym_controllers import OpenAIGymMainController
from corl.simulators.openai_gym.gym_sensors import OpenAiGymStateSensor
from corl.simulators.openai_gym.gym_simulator import OpenAiGymPlatform

def test_override_controller_sensor_prop():
    NUM_DISCRETE = 3
    base_environment = gym.make("CartPole-v1")

    # this is a discrete
    base_action_space = base_environment.action_space
    base_obs_space = base_environment.observation_space

    controller_mod_part = (OpenAIGymMainController, {"properties": {
        "n": NUM_DISCRETE
    }})
    sensor_mod_part = (OpenAiGymStateSensor, {"properties": {
        "high": list(base_obs_space.high / 2),
        "low": list(base_obs_space.low / 2)
    }})
    platform_mod_parts = [controller_mod_part, sensor_mod_part]

    openai_mod_platform = OpenAiGymPlatform(platform_name="blue0", platform=base_environment, parts_list=platform_mod_parts)

    controller_space = openai_mod_platform.controllers[0].control_properties.create_space()
    sensor_space = openai_mod_platform.sensors[0].measurement_properties.create_space()

    assert controller_space != base_action_space
    assert controller_space == gym.spaces.Discrete(NUM_DISCRETE)
    assert sensor_space != base_obs_space





