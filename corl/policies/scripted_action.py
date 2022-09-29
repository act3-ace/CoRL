"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Scripted Action Policy
"""
import typing

import flatten_dict
import numpy as np
from pydantic import validator

from corl.libraries.env_space_util import EnvSpaceUtil
from corl.policies.base_policy import BasePolicyValidator
from corl.policies.custom_policy import CustomPolicy, CustomPolicyValidator


class ScriptedActionPolicyValidator(CustomPolicyValidator):
    """Validator for the ScriptedActionPolicy"""

    control_times: typing.List[float]
    control_values: typing.List[typing.Dict]

    missing_action_policy: typing.Literal['default_action', 'repeat_last_action']
    default_action: typing.Optional[typing.Dict] = None

    class Config:
        """pydantic configuration options"""
        arbitrary_types_allowed = True

    @validator('control_times')
    def sort_control_times(cls, v):  # pylint: disable=no-self-argument, no-self-use
        """Ensures that control_times are in order"""
        assert v == sorted(v), "control_times must be in order"
        return v

    @staticmethod
    def convert_control_value(controls, controller_key_paths, sample_control):
        """converts the controls into a dict"""

        flat_sample_control = flatten_dict.flatten(sample_control)

        assert len(controls) == len(controller_key_paths), 'mismatch between number of controllers and length of control values'
        assert len(controller_key_paths) == len(flat_sample_control), 'mismatch between number of controllers and the action_space'

        flat_control_dict = {}
        for i, ctrl_value in enumerate(controls):
            try:
                controller_key = controller_key_paths[i]
                sample_value = flat_sample_control[controller_key]
                if isinstance(sample_value, np.ndarray):
                    flat_control_dict[controller_key] = np.add(sample_value * 0, ctrl_value, dtype=sample_value.dtype)
                else:
                    flat_control_dict[controller_key] = type(sample_value)(ctrl_value)
            except Exception as e:
                raise RuntimeError(f'@idx: {i}, controller: {controller_key}, control_value: {ctrl_value}') from e
        control_dict = flatten_dict.unflatten(flat_control_dict)
        return control_dict

    @staticmethod
    def convert_control_values(action_space, controller_key_paths, control_times, controls_list):
        """converts the input_control values into dictionary and validates it agaist the action space"""
        assert len(controls_list) == len(control_times), 'mismatch between number of control_times and control_values'

        sample_control = action_space.sample()

        converted_control_list: typing.List[typing.Dict] = []
        for controls in controls_list:

            control_dict = ScriptedActionPolicyValidator.convert_control_value(controls, controller_key_paths, sample_control)

            EnvSpaceUtil.deep_sanity_check_space_sample(action_space, control_dict)

            converted_control_list.append(control_dict)

        return converted_control_list

    @validator('control_values', pre=True, always=True)
    def validate_control_values(cls, controls_list, values):  # pylint: disable=no-self-argument, no-self-use
        """validate that control_values match the controllers"""
        if 'controllers' not in values or 'control_times' not in values:
            raise ValueError(f'Could not run "validate_control_values" because previous items failed: {values}')

        return ScriptedActionPolicyValidator.convert_control_values(
            values['act_space'], values['controllers'], values['control_times'], controls_list
        )

    @validator('missing_action_policy')
    def validate_missing_action_policy(cls, missing_action_policy, values):
        """validates the missing action policy"""
        if missing_action_policy == 'repeat_last_action':
            assert values['control_times'][0] == 0, "missing control_time for t=0"
        return missing_action_policy

    @validator('default_action', pre=True)
    def validate_default_action(cls, default_action, values):
        """validates that the default_action is consistent with the missing_action_policy"""
        if values["missing_action_policy"] == 'default_action':
            assert default_action is not None, 'default_action is requried when using the default_action missing_action_policy'

            action_space = values['act_space']
            action = ScriptedActionPolicyValidator.convert_control_value(default_action, values['controllers'], action_space.sample())

            return action

        assert default_action is None, 'default_action is invalid except when using the default_action missing_action_policy'
        return default_action


class ScriptedActionPolicy(CustomPolicy):  # pylint: disable=abstract-method
    """Scripted action policy.
    """

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self._input_index: int
        self._last_action: dict

    @property
    def get_validator(self) -> typing.Type[BasePolicyValidator]:
        """
        Get the validator for this experiment class,
        the kwargs sent to the experiment class will
        be validated using this object and add a self.config
        attr to the experiment class
        """
        return ScriptedActionPolicyValidator

    def _reset(self):
        super()._reset()
        self._input_index = 0
        self._last_action = EnvSpaceUtil.get_zero_sample_from_space(self.validated_config.act_space)

    def custom_compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        sim_time=None,
        agent_id=None,
        info=None,
        episode=None,
        **kwargs
    ):
        for control_index in range(self._input_index, len(self.validated_config.control_times)):
            control_time = self.validated_config.control_times[control_index]
            if sim_time >= control_time:
                # apply control_list to controls
                control_values = self.validated_config.control_values[control_index]

                self._input_index = control_index + 1
                self._last_action = control_values
                return [control_values], [], {}

            break

        if self.validated_config.missing_action_policy == 'repeat_last_action':
            return [self._last_action], [], {}

        self._last_action = self.validated_config.default_action
        return [self.validated_config.default_action], [], {}
