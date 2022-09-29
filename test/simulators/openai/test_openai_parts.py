import gym
from corl.simulators.openai_gym.gym_controllers import OpenAIGymMainController
from corl.simulators.openai_gym.gym_sensors import OpenAiGymStateSensor
from corl.simulators.openai_gym.gym_simulator import OpenAiGymPlatform

def test_openai_controller_prop():
    base_environment = gym.make("CartPole-v1")

    # this is a discrete
    base_action_space = base_environment.action_space
    base_obs_space = base_environment.observation_space

    controller_part = (OpenAIGymMainController, {})
    sensor_part = (OpenAiGymStateSensor, {})
    base_platform_parts = [controller_part, sensor_part]

    openai_platform = OpenAiGymPlatform(platform_name="blue0", platform=base_environment, parts_list=base_platform_parts)

    assert openai_platform.controllers[0].control_properties.create_space() == base_action_space
    assert openai_platform.sensors[0].measurement_properties.create_space() == base_obs_space