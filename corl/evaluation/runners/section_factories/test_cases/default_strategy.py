"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
This module defines the TestCaseStrategy for running evaluation with the default EPPs defined in the task configs.

Auther: John McCarroll
"""

import typing

from pydantic import BaseModel

from corl.evaluation.runners.section_factories.test_cases.test_case_manager import TestCaseStrategy


class IncrementalParameterProviderValidator(BaseModel):
    """
    A validator for DefaultStrategy class
    """
    num_test_cases: int = 1


class DefaultStrategy(TestCaseStrategy):
    """
    The concrete implementation of a TestCaseStrategy for a task's default ParameterProviders.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.incremental_epp_config: IncrementalParameterProviderValidator = self.get_incremental_epp_validator(
            **self.config.config
        )  # type: ignore
        self.num_test_cases = self.incremental_epp_config.num_test_cases
        self.epp_wrapper_path = "corl.episode_parameter_providers.incremental.IncrementalParameterProviderWrapper"

    @property
    def get_incremental_epp_validator(self) -> typing.Type[IncrementalParameterProviderValidator]:
        """
        Method to return validator for IncrementalParameterProviderValidator
        """
        return IncrementalParameterProviderValidator

    def update_rllib_config(self, config: dict):
        """
        Method responsible for mutating rllib config, overwriting the task's default EPPs with TabularParameterProviders.
        """
        # wrap EPPs w IncrementalParameterProviderWrapper

        # retreive epp config for environment's params
        # pass epp config to wrapper
        epp_env_type = config['env_config']['episode_parameter_provider']['type']
        wrapper_config = {
            "num_test_cases": self.num_test_cases,
            "type": epp_env_type,
        }
        # add config, if available
        epp_env_config = config['env_config']['episode_parameter_provider']['config'] if 'config' in config['env_config'][
            'episode_parameter_provider'] else None
        if epp_env_config:
            wrapper_config['config'] = epp_env_config

        config['env_config']['episode_parameter_provider'] = {'type': self.epp_wrapper_path, 'config': wrapper_config}

        # Wrap the agents EPPs w Incremental EPP
        for _, value in config['env_config']['agents'].items():

            # retreive epp config for agent's params
            epp_agent_type = value.class_config.config['episode_parameter_provider']['type']
            wrapper_config = {
                "num_test_cases": self.num_test_cases,
                "type": epp_agent_type,
            }
            # add config, if aailable
            epp_agent_config = value.class_config.config['episode_parameter_provider']['config'] if 'config' in value.class_config.config[
                'episode_parameter_provider'] else None
            if epp_agent_config:
                wrapper_config['config'] = epp_env_config

            value.class_config.config['episode_parameter_provider'] = {'type': self.epp_wrapper_path, 'config': wrapper_config}

        return config

    def get_test_cases(self):
        """
        Method responsible for retreiving test cases planned for evaluation.
        Using default strategy does not plan test cases by a collection of parameter values.
        """
        return None

    def get_num_test_cases(self) -> int:
        """
        Method responsible for reporting the number of test cases planned for evaluation.
        """
        return self.num_test_cases
