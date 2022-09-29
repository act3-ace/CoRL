"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import typing

import pytest

from corl.agents.base_agent import AgentParseBase, TrainableBaseAgent
from corl.simulators.openai_gym.gym_available_platforms import OpenAIGymAvailablePlatformTypes
from corl.simulators.openai_gym.gym_simulator import OpenAIGymSimulator
from corl.simulators.openai_gym.gym_controllers import OpenAIGymMainController
from corl.simulators.openai_gym.gym_sensors import OpenAiGymStateSensor
from corl.simulators.base_parts import BasePlatformPartValidator
from corl.libraries.plugin_library import PluginLibrary


# These controller and sensor classes add an additional parameter to their validator.  The purpose
# of the tests are to ensure that this parameter is properly passed from the reference store into
# the constructor of the class.  As such, the parameter is added to the validator but is never
# actually used in the controller/sensor.  A real implementation would only add a parameter because
# it had a purpose such that the parameter was in some way referenced by the class.
class ControllerArgsValidator(BasePlatformPartValidator):
    param_controller: int


class ControllerArgs(OpenAIGymMainController):
    @property
    def get_validator(self) -> typing.Type[ControllerArgsValidator]:
        return ControllerArgsValidator


PluginLibrary.AddClassToGroup(
    ControllerArgs, "Controller_Gym_Args", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


class SensorArgsValidator(BasePlatformPartValidator):
    param_sensor: int


class SensorArgs(OpenAiGymStateSensor):
    @property
    def get_validator(self) -> typing.Type[SensorArgsValidator]:
        return SensorArgsValidator


PluginLibrary.AddClassToGroup(
    SensorArgs, "Sensor_State_Args", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


def test_valid():

    agent_id = 'blue0'

    controller_arg_value = 42
    sensor_arg_value = 45434

    config = {
        'parts': [
            {
                'part': 'Controller_Gym_Args',
                'references': {
                    'param_controller': 'param_controller_source'
                }
            },
            {
                'part': 'Sensor_State_Args',
                'references': {
                    'param_sensor': 'param_sensor_source'
                }
            }
        ],
        'episode_parameter_provider': {
            "type": "corl.episode_parameter_providers.simple.SimpleParameterProvider"
        },
        'glues': [
            {
                "functor": "corl.glues.common.controller_glue.ControllerGlue",
                "config": {
                    "controller": "Controller_Gym_Args",
                },
            },
            {
                "functor": "corl.glues.common.observe_sensor.ObserveSensor",
                "config":{
                    "sensor": "Sensor_State_Args",
                    "normalization": {
                    "enabled": False
                    }
                }
            },
        ],
        'dones': [
            {
                "functor": "corl.dones.openai_gym_done.OpenAIGymDone",
                "config": {},
            },
        ],
        'rewards': [
            {
                "name": "OpenAIGymReward",
                "functor": "corl.rewards.openai_gym_reward.OpenAIGymReward",
                "config": {},
            },
        ],
        'reference_store': {
            'param_controller_source': controller_arg_value,
            'param_sensor_source': sensor_arg_value
        }
    }
    agent_parse_base = AgentParseBase(agent=TrainableBaseAgent, config=config)

    agent_class = agent_parse_base.agent(
        **agent_parse_base.config, agent_name='foobar', platform_name=agent_id
    )
    parts = agent_class.get_platform_parts(OpenAIGymSimulator, OpenAIGymAvailablePlatformTypes.MAIN)

    agent_configs = {
        agent_id: {
            'platform_config': {
                'platform_class': 'corl.simulators.openai_gym.gym_simulator.OpenAiGymPlatform'
            },
            'parts_list': parts
        }
    }

    sim_reset_config = {
        'platforms': {agent_id: {}},
        'agent_configs_reset': agent_configs
    }

    simulator = OpenAIGymSimulator(gym_env='CartPole-v1', seed=1, agent_configs=agent_configs)

    # If this test is broken, the following line will raise a pydantic.ValidationError because
    # `param_controller` and `param_sensor` are not properly passed from the reference store into
    # the necessary constructor.
    simulator.reset(sim_reset_config)

    assert simulator.sim_platforms[0].controllers[0].config.param_controller == controller_arg_value
    assert simulator.sim_platforms[0].sensors[0].config.param_sensor == sensor_arg_value


def test_missing_reference():
    # This differs from test_valid in that the param_controller_source is missing from the
    # reference store.  Therefore, it throws a RuntimeError.

    agent_id = 'blue0'

    controller_arg_value = 42
    sensor_arg_value = 45434

    config = {
        'parts': [
            {
                'part': 'Controller_Gym_Args',
                'references': {
                    'param_controller': 'param_controller_source'
                }
            },
            {
                'part': 'Sensor_State_Args',
                'references': {
                    'param_sensor': 'param_sensor_source'
                }
            }
        ],
        'episode_parameter_provider': {
            "type": "corl.episode_parameter_providers.simple.SimpleParameterProvider"
        },
        'glues': [
            {
                "functor": "corl.glues.common.controller_glue.ControllerGlue",
                "config": {
                    "controller": "Controller_Gym_Args",
                },
            },
            {
                "functor": "corl.glues.common.observe_sensor.ObserveSensor",
                "config":{
                    "sensor": "Sensor_State_Args",
                    "normalization": {
                    "enabled": False
                    }
                }
            },
        ],
        'dones': [
            {
                "functor": "corl.dones.openai_gym_done.OpenAIGymDone",
                "config": {},
            },
        ],
        'rewards': [
            {
                "name": "OpenAIGymReward",
                "functor": "corl.rewards.openai_gym_reward.OpenAIGymReward",
                "config": {},
            },
        ],
        'reference_store': {
            'param_sensor_source': sensor_arg_value
        }
    }
    agent_parse_base = AgentParseBase(agent=TrainableBaseAgent, config=config)

    agent_class = agent_parse_base.agent(
        **agent_parse_base.config, agent_name='foobar', platform_name=agent_id
    )

    with pytest.raises(RuntimeError):
        parts = agent_class.get_platform_parts(OpenAIGymSimulator, OpenAIGymAvailablePlatformTypes.MAIN)


def test_parameter():
    # This differs from test_valid in that the param_controller_source is a Parameter.  This is not
    # allowed because the platform parts are not recreated in the environment reset.  Therefore, it
    # raises a TypeError.

    agent_id = 'blue0'

    controller_arg_value = 42
    sensor_arg_value = 45434

    config = {
        'parts': [
            {
                'part': 'Controller_Gym_Args',
                'references': {
                    'param_controller': 'param_controller_source'
                }
            },
            {
                'part': 'Sensor_State_Args',
                'references': {
                    'param_sensor': 'param_sensor_source'
                }
            }
        ],
        'episode_parameter_provider': {
            "type": "corl.episode_parameter_providers.simple.SimpleParameterProvider"
        },
        'glues': [
            {
                "functor": "corl.glues.common.controller_glue.ControllerGlue",
                "config": {
                    "controller": "Controller_Gym_Args",
                },
            },
            {
                "functor": "corl.glues.common.observe_sensor.ObserveSensor",
                "config":{
                    "sensor": "Sensor_State_Args",
                    "normalization": {
                    "enabled": False
                    }
                }
            },
        ],
        'dones': [
            {
                "functor": "corl.dones.openai_gym_done.OpenAIGymDone",
                "config": {},
            },
        ],
        'rewards': [
            {
                "name": "OpenAIGymReward",
                "functor": "corl.rewards.openai_gym_reward.OpenAIGymReward",
                "config": {},
            },
        ],
        'reference_store': {
            'param_controller_source': {
                'type': 'corl.libraries.parameters.ConstantParameter',
                'config': {
                    'units': None,
                    'value': controller_arg_value
                }
            },
            'param_sensor_source': sensor_arg_value
        }
    }
    agent_parse_base = AgentParseBase(agent=TrainableBaseAgent, config=config)

    agent_class = agent_parse_base.agent(
        **agent_parse_base.config, agent_name='foobar', platform_name=agent_id
    )

    with pytest.raises(TypeError):
        parts = agent_class.get_platform_parts(OpenAIGymSimulator, OpenAIGymAvailablePlatformTypes.MAIN)
