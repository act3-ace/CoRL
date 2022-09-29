"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import abc
import typing
from collections import OrderedDict

import gym
import pytest
from pydantic import BaseModel

from corl.agents.base_agent import AgentParseBase, TrainableBaseAgent
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from corl.rewards.reward_func_wrapper import BaseWrapperReward, BaseWrapperRewardValidator
from corl.rewards.reward_func_multi_wrapper import BaseMultiWrapperReward, BaseMultiWrapperRewardValidator
from corl.rewards.reward_func_dict_wrapper import BaseDictWrapperReward, BaseDictWrapperRewardValidator
from corl.libraries.factory import Factory

# These tests look at the interplay between functors (including the various functor wrappers) and parameters/references.  It uses rewards
# because they are easier to create than glues (no simulator or platforms required); however, it would work for any object created from
#  functors, including glues, rewards, and dones.

# All of the parameters are very intentionally called 'param1' and 'param2' to ensure that there are no name conflicts between parameters
# with the same name in different classes.


class NormalRewardValidator(RewardFuncBaseValidator):
    param1: float
    param2: float


class NormalReward(RewardFuncBase):
    @property
    def get_validator(self) -> typing.Type[NormalRewardValidator]:
        return NormalRewardValidator


class WrappingRewardValidator(BaseWrapperRewardValidator):
    param1: float
    param2: float


class WrappingReward(BaseWrapperReward):
    @property
    def get_validator(self) -> typing.Type[WrappingRewardValidator]:
        return WrappingRewardValidator


class MultiWrappingRewardValidator(BaseMultiWrapperRewardValidator):
    param1: float
    param2: float


class MultiWrappingReward(BaseMultiWrapperReward):
    @property
    def get_validator(self) -> typing.Type[MultiWrappingRewardValidator]:
        return MultiWrappingRewardValidator


class DictWrappingRewardValidator(BaseDictWrapperRewardValidator):
    param1: float
    param2: float


class DictWrappingReward(BaseDictWrapperReward):
    @property
    def get_validator(self) -> typing.Type[DictWrappingRewardValidator]:
        return DictWrappingRewardValidator


def _get_base_config():

    return {
        'parts': [
            {
                'part': 'Controller_Gym',
            },
            {
                'part': 'Sensor_State',
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
            }
        ],
        'reference_store': {}
    }


def test_functor_parameters_references():

    ##### Setup #####
    agent_id = 'blue0'

    config = _get_base_config()

    # Add a "normal" reward with parameters and references
    config['rewards'].append(
        {
            'functor': NormalReward,
            'config': {
                'param1': {
                    'type': 'corl.libraries.parameters.UniformParameter',
                    'config': {
                        'low': 0,
                        'high': 1,
                        'units': None
                    }
                },
            },
            'references': {
                'param2': 'normal_param'
            }
        }
    )
    config['reference_store']['normal_param'] = {
        'type': 'corl.libraries.parameters.UniformParameter',
        'config': {
            'low': 2,
            'high': 3,
            'units': None
        }
    }

    agent_parse_base = AgentParseBase(agent=TrainableBaseAgent, config=config)

    agent_class = agent_parse_base.agent(
        **agent_parse_base.config, agent_name='foobar', platform_name=agent_id
    )

    # Pairs of identical random number generators
    rng1a, _ = gym.utils.seeding.np_random(0)
    rng1b, _ = gym.utils.seeding.np_random(0)
    rng2a, _ = gym.utils.seeding.np_random(314)
    rng2b, _ = gym.utils.seeding.np_random(314)

    # Build the factories for the parameters so that they can be sampled for comparison
    factory1 = Factory(**config['rewards'][-1]['config']['param1'])
    param1_parameter = factory1.build()

    factory2 = Factory(**config['reference_store']['normal_param'])
    param2_parameter = factory2.build()

    ##### Default Parameters #####

    # Fill the local variable store from a known random number generator
    # Make the rewards from the default parameters
    agent_class.fill_parameters(rng1a, default_parameters=True)
    agent_class.make_rewards(agent_id, [])

    # Determine what value is randomly sampled from the default parameters by using the same random number generator
    param1_value = param1_parameter.get_value(rng1b).value
    param2_value = param2_parameter.get_value(rng1b).value

    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param1 == pytest.approx(param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param2 == pytest.approx(param2_value)

    ##### Parameters that come from EPP #####
    # This part makes sure that the EPP produces the parameters with different values from the defaults.

    agent_class.fill_parameters(rng2a, default_parameters=False)
    agent_class.make_rewards(agent_id, [])

    param1_value = param1_parameter.get_value(rng2b).value
    param2_value = param2_parameter.get_value(rng2b).value

    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param1 == pytest.approx(param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param2 == pytest.approx(param2_value)    


def test_functor_wrapper_parameters_references():

    ##### Setup #####
    agent_id = 'blue0'

    config = _get_base_config()

    # Add a wrapping reward around a normal reward
    config['rewards'].append(
        {
            'functor': WrappingReward,
            'config': {
                'param1': {
                    'type': 'corl.libraries.parameters.UniformParameter',
                    'config': {
                        'low': 4,
                        'high': 5,
                        'units': None
                    }
                },
            },
            'references': {
                'param2': 'wrapping_param'
            },
            'wrapped': {
                'functor': NormalReward,
                'config': {
                    'param1': {
                        'type': 'corl.libraries.parameters.UniformParameter',
                        'config': {
                            'low': 0,
                            'high': 1,
                            'units': None
                        }
                    }
                },
                'references': {
                    'param2': 'normal_param'
                }
            }
        }
    )
    config['reference_store']['normal_param'] = {
        'type': 'corl.libraries.parameters.UniformParameter',
        'config': {
            'low': 2,
            'high': 3,
            'units': None
        }
    }
    config['reference_store']['wrapping_param'] = {
        'type': 'corl.libraries.parameters.UniformParameter',
        'config': {
            'low': 6,
            'high': 7,
            'units': None
        }
    }

    agent_parse_base = AgentParseBase(agent=TrainableBaseAgent, config=config)

    agent_class = agent_parse_base.agent(
        **agent_parse_base.config, agent_name='foobar', platform_name=agent_id
    )

    # Pairs of identical random number generators
    rng1a, _ = gym.utils.seeding.np_random(0)
    rng1b, _ = gym.utils.seeding.np_random(0)
    rng2a, _ = gym.utils.seeding.np_random(314)
    rng2b, _ = gym.utils.seeding.np_random(314)

    # Build the factories for the parameters so that they can be sampled for comparison
    factory1 = Factory(**config['rewards'][-1]['wrapped']['config']['param1'])
    internal_param1_parameter = factory1.build()

    factory2 = Factory(**config['reference_store']['normal_param'])
    internal_param2_parameter = factory2.build()

    factory3 = Factory(**config['rewards'][-1]['config']['param1'])
    wrapping_param1_parameter = factory3.build()

    factory4 = Factory(**config['reference_store']['wrapping_param'])
    wrapping_param2_parameter = factory4.build()

    ##### Default Parameters #####

    # Fill the local variable store from a known random number generator
    # Make the rewards from the default parameters
    agent_class.fill_parameters(rng1a, default_parameters=True)
    agent_class.make_rewards(agent_id, [])

    # Determine what value is randomly sampled from the default parameters by using the same random number generator
    # The order of these must match how the agent fills parameters.
    wrapping_param1_value = wrapping_param1_parameter.get_value(rng1b).value
    internal_param1_value = internal_param1_parameter.get_value(rng1b).value
    internal_param2_value = internal_param2_parameter.get_value(rng1b).value
    wrapping_param2_value = wrapping_param2_parameter.get_value(rng1b).value

    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param1 == pytest.approx(wrapping_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param2 == pytest.approx(wrapping_param2_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped.config.param1 == pytest.approx(internal_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped.config.param2 == pytest.approx(internal_param2_value)

    ##### Parameters that come from EPP #####
    # This part makes sure that the EPP produces the parameters with different values from the defaults.

    agent_class.fill_parameters(rng2a, default_parameters=False)
    agent_class.make_rewards(agent_id, [])
    
    wrapping_param1_value = wrapping_param1_parameter.get_value(rng2b).value
    internal_param1_value = internal_param1_parameter.get_value(rng2b).value
    internal_param2_value = internal_param2_parameter.get_value(rng2b).value
    wrapping_param2_value = wrapping_param2_parameter.get_value(rng2b).value
    
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param1 == pytest.approx(wrapping_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param2 == pytest.approx(wrapping_param2_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped.config.param1 == pytest.approx(internal_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped.config.param2 == pytest.approx(internal_param2_value)  


def test_functor_multi_wrapper_parameters_references():

    ##### Setup #####
    agent_id = 'blue0'

    config = _get_base_config()

    # Place multiple normal rewards within a multi-wrapping reward.
    config['rewards'].append(
        {
            'functor': MultiWrappingReward,
            'config': {
                'param1': {
                    'type': 'corl.libraries.parameters.UniformParameter',
                    'config': {
                        'low': 4,
                        'high': 5,
                        'units': None
                    }
                },
            },
            'references': {
                'param2': 'wrapping_param'
            },
            'wrapped': [
                {
                    'name': 'first',
                    'functor': NormalReward,
                    'config': {
                        'param1': {
                            'type': 'corl.libraries.parameters.UniformParameter',
                            'config': {
                                'low': 0,
                                'high': 1,
                                'units': None
                            }
                        }
                    },
                    'references': {
                        'param2': 'first_normal_param'
                    }
                },
                {
                    'name': 'second',
                    'functor': NormalReward,
                    'config': {
                        'param1': {
                            'type': 'corl.libraries.parameters.UniformParameter',
                            'config': {
                                'low': 8,
                                'high': 9,
                                'units': None
                            }
                        }
                    },
                    'references': {
                        'param2': 'second_normal_param'
                    }
                }
            ]
        }
    )
    config['reference_store']['first_normal_param'] = {
        'type': 'corl.libraries.parameters.UniformParameter',
        'config': {
            'low': 2,
            'high': 3,
            'units': None
        }
    }
    config['reference_store']['second_normal_param'] = {
        'type': 'corl.libraries.parameters.UniformParameter',
        'config': {
            'low': 10,
            'high': 11,
            'units': None
        }
    }
    config['reference_store']['wrapping_param'] = {
        'type': 'corl.libraries.parameters.UniformParameter',
        'config': {
            'low': 6,
            'high': 7,
            'units': None
        }
    }

    agent_parse_base = AgentParseBase(agent=TrainableBaseAgent, config=config)

    agent_class = agent_parse_base.agent(
        **agent_parse_base.config, agent_name='foobar', platform_name=agent_id
    )

    # Pairs of identical random number generators
    rng1a, _ = gym.utils.seeding.np_random(0)
    rng1b, _ = gym.utils.seeding.np_random(0)
    rng2a, _ = gym.utils.seeding.np_random(314)
    rng2b, _ = gym.utils.seeding.np_random(314)

    # Build the factories for the parameters so that they can be sampled for comparison
    factory1 = Factory(**config['rewards'][-1]['wrapped'][0]['config']['param1'])
    first_internal_param1_parameter = factory1.build()

    factory2 = Factory(**config['reference_store']['first_normal_param'])
    first_internal_param2_parameter = factory2.build()

    factory1b = Factory(**config['rewards'][-1]['wrapped'][1]['config']['param1'])
    second_internal_param1_parameter = factory1b.build()

    factory2b = Factory(**config['reference_store']['second_normal_param'])
    second_internal_param2_parameter = factory2b.build()

    factory3 = Factory(**config['rewards'][-1]['config']['param1'])
    wrapping_param1_parameter = factory3.build()

    factory4 = Factory(**config['reference_store']['wrapping_param'])
    wrapping_param2_parameter = factory4.build()

    ##### Default Parameters #####

    # Fill the local variable store from a known random number generator
    # Make the rewards from the default parameters
    agent_class.fill_parameters(rng1a, default_parameters=True)
    agent_class.make_rewards(agent_id, [])

    # Determine what value is randomly sampled from the default parameters by using the same random number generator
    # The order of these must match how the agent fills parameters.
    wrapping_param1_value = wrapping_param1_parameter.get_value(rng1b).value
    first_internal_param1_value = first_internal_param1_parameter.get_value(rng1b).value
    second_internal_param1_value = second_internal_param1_parameter.get_value(rng1b).value
    first_internal_param2_value = first_internal_param2_parameter.get_value(rng1b).value
    second_internal_param2_value = second_internal_param2_parameter.get_value(rng1b).value
    wrapping_param2_value = wrapping_param2_parameter.get_value(rng1b).value

    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param1 == pytest.approx(wrapping_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param2 == pytest.approx(wrapping_param2_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped[0].config.param1 == pytest.approx(first_internal_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped[0].config.param2 == pytest.approx(first_internal_param2_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped[1].config.param1 == pytest.approx(second_internal_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped[1].config.param2 == pytest.approx(second_internal_param2_value)

    ##### Parameters that come from EPP #####
    # This part makes sure that the EPP produces the parameters with different values from the defaults.

    agent_class.fill_parameters(rng2a, default_parameters=False)
    agent_class.make_rewards(agent_id, [])
    
    wrapping_param1_value = wrapping_param1_parameter.get_value(rng2b).value
    first_internal_param1_value = first_internal_param1_parameter.get_value(rng2b).value
    second_internal_param1_value = second_internal_param1_parameter.get_value(rng2b).value
    first_internal_param2_value = first_internal_param2_parameter.get_value(rng2b).value
    second_internal_param2_value = second_internal_param2_parameter.get_value(rng2b).value
    wrapping_param2_value = wrapping_param2_parameter.get_value(rng2b).value
    
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param1 == pytest.approx(wrapping_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param2 == pytest.approx(wrapping_param2_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped[0].config.param1 == pytest.approx(first_internal_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped[0].config.param2 == pytest.approx(first_internal_param2_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped[1].config.param1 == pytest.approx(second_internal_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped[1].config.param2 == pytest.approx(second_internal_param2_value) 


def test_functor_dict_wrapper_parameters_references():

    ##### Setup #####
    agent_id = 'blue0'

    config = _get_base_config()

    # Place multiple normal rewards within a dictionary wrapping reward.
    config['rewards'].append(
        {
            'functor': DictWrappingReward,
            'config': {
                'param1': {
                    'type': 'corl.libraries.parameters.UniformParameter',
                    'config': {
                        'low': 4,
                        'high': 5,
                        'units': None
                    }
                },
            },
            'references': {
                'param2': 'wrapping_param'
            },
            'wrapped': {
                'first': {
                    'functor': NormalReward,
                    'config': {
                        'param1': {
                            'type': 'corl.libraries.parameters.UniformParameter',
                            'config': {
                                'low': 0,
                                'high': 1,
                                'units': None
                            }
                        }
                    },
                    'references': {
                        'param2': 'first_normal_param'
                    }
                },
                'second': {
                    'functor': NormalReward,
                    'config': {
                        'param1': {
                            'type': 'corl.libraries.parameters.UniformParameter',
                            'config': {
                                'low': 8,
                                'high': 9,
                                'units': None
                            }
                        }
                    },
                    'references': {
                        'param2': 'second_normal_param'
                    }
                }
            }
        }
    )
    config['reference_store']['first_normal_param'] = {
        'type': 'corl.libraries.parameters.UniformParameter',
        'config': {
            'low': 2,
            'high': 3,
            'units': None
        }
    }
    config['reference_store']['second_normal_param'] = {
        'type': 'corl.libraries.parameters.UniformParameter',
        'config': {
            'low': 10,
            'high': 11,
            'units': None
        }
    }
    config['reference_store']['wrapping_param'] = {
        'type': 'corl.libraries.parameters.UniformParameter',
        'config': {
            'low': 6,
            'high': 7,
            'units': None
        }
    }

    agent_parse_base = AgentParseBase(agent=TrainableBaseAgent, config=config)

    agent_class = agent_parse_base.agent(
        **agent_parse_base.config, agent_name='foobar', platform_name=agent_id
    )

    # Pairs of identical random number generators
    rng1a, _ = gym.utils.seeding.np_random(0)
    rng1b, _ = gym.utils.seeding.np_random(0)
    rng2a, _ = gym.utils.seeding.np_random(314)
    rng2b, _ = gym.utils.seeding.np_random(314)

    # Build the factories for the parameters so that they can be sampled for comparison
    factory1 = Factory(**config['rewards'][-1]['wrapped']['first']['config']['param1'])
    first_internal_param1_parameter = factory1.build()

    factory2 = Factory(**config['reference_store']['first_normal_param'])
    first_internal_param2_parameter = factory2.build()

    factory1b = Factory(**config['rewards'][-1]['wrapped']['second']['config']['param1'])
    second_internal_param1_parameter = factory1b.build()

    factory2b = Factory(**config['reference_store']['second_normal_param'])
    second_internal_param2_parameter = factory2b.build()

    factory3 = Factory(**config['rewards'][-1]['config']['param1'])
    wrapping_param1_parameter = factory3.build()

    factory4 = Factory(**config['reference_store']['wrapping_param'])
    wrapping_param2_parameter = factory4.build()

    ##### Default Parameters #####

    # Fill the local variable store from a known random number generator
    # Make the rewards from the default parameters
    agent_class.fill_parameters(rng1a, default_parameters=True)
    agent_class.make_rewards(agent_id, [])

    # Determine what value is randomly sampled from the default parameters by using the same random number generator
    # The order of these must match how the agent fills parameters.
    wrapping_param1_value = wrapping_param1_parameter.get_value(rng1b).value
    first_internal_param1_value = first_internal_param1_parameter.get_value(rng1b).value
    second_internal_param1_value = second_internal_param1_parameter.get_value(rng1b).value
    first_internal_param2_value = first_internal_param2_parameter.get_value(rng1b).value
    second_internal_param2_value = second_internal_param2_parameter.get_value(rng1b).value
    wrapping_param2_value = wrapping_param2_parameter.get_value(rng1b).value

    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param1 == pytest.approx(wrapping_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param2 == pytest.approx(wrapping_param2_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped['first'].config.param1 == pytest.approx(first_internal_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped['first'].config.param2 == pytest.approx(first_internal_param2_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped['second'].config.param1 == pytest.approx(second_internal_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped['second'].config.param2 == pytest.approx(second_internal_param2_value)

    ##### Parameters that come from EPP #####
    # This part makes sure that the EPP produces the parameters with different values from the defaults.

    agent_class.fill_parameters(rng2a, default_parameters=False)
    agent_class.make_rewards(agent_id, [])
    
    wrapping_param1_value = wrapping_param1_parameter.get_value(rng2b).value
    first_internal_param1_value = first_internal_param1_parameter.get_value(rng2b).value
    second_internal_param1_value = second_internal_param1_parameter.get_value(rng2b).value
    first_internal_param2_value = first_internal_param2_parameter.get_value(rng2b).value
    second_internal_param2_value = second_internal_param2_parameter.get_value(rng2b).value
    wrapping_param2_value = wrapping_param2_parameter.get_value(rng2b).value
    
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param1 == pytest.approx(wrapping_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.param2 == pytest.approx(wrapping_param2_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped['first'].config.param1 == pytest.approx(first_internal_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped['first'].config.param2 == pytest.approx(first_internal_param2_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped['second'].config.param1 == pytest.approx(second_internal_param1_value)
    assert agent_class.agent_reward_dict.process_callbacks[-1].config.wrapped['second'].config.param2 == pytest.approx(second_internal_param2_value)
