"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
This module defines the TestCaseStrategy for running evaluation with the default EPPs defined in the task configs.

Author: John McCarroll
"""


from corl.evaluation.runners.section_factories.test_cases.test_case_manager import (
    TestCaseIndex,
    TestCaseStrategy,
    TestCaseStrategyValidator,
)


class DefaultStrategyValidator(TestCaseStrategyValidator):
    """
    A validator for DefaultStrategy class
    """

    num_test_cases: int = 1
    epp_wrapper_path: str = "corl.episode_parameter_providers.incremental.IncrementalParameterProviderWrapper"


class DefaultStrategy(TestCaseStrategy):
    """
    The concrete implementation of a TestCaseStrategy for a task's default ParameterProviders.
    """

    def __init__(self, **kwargs) -> None:
        self.config: DefaultStrategyValidator
        self.test_cases: list = []
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[DefaultStrategyValidator]:
        """
        Method to return validator for DefaultStrategyValidator
        """
        return DefaultStrategyValidator

    def update_rllib_config(self, config: dict):
        """
        Method responsible for mutating rllib config, overwriting the task's default EPPs with TabularParameterProviders.
        """
        # wrap EPPs w IncrementalParameterProviderWrapper

        # retrieve epp config for environment's params
        # pass epp config to wrapper
        epp_env_type = config["env_config"]["episode_parameter_provider"]["type"]
        wrapper_config = {
            "num_test_cases": self.config.num_test_cases,
            "type": epp_env_type,
        }
        # add config, if available
        epp_env_config = (
            config["env_config"]["episode_parameter_provider"]["config"]
            if "config" in config["env_config"]["episode_parameter_provider"]
            else None
        )
        if epp_env_config:
            wrapper_config["config"] = epp_env_config

        config["env_config"]["episode_parameter_provider"] = {"type": self.config.epp_wrapper_path, "config": wrapper_config}

        # Wrap the agents EPPs w Incremental EPP
        for value in config["env_config"]["agents"].values():
            # retrieve epp config for agent's params
            epp_agent_type = value.class_config.config["episode_parameter_provider"]["type"]
            wrapper_config = {
                "num_test_cases": self.config.num_test_cases,
                "type": epp_agent_type,
            }
            # add config, if aailable
            epp_agent_config = (
                value.class_config.config["episode_parameter_provider"]["config"]
                if "config" in value.class_config.config["episode_parameter_provider"]
                else None
            )
            if epp_agent_config:
                wrapper_config["config"] = epp_env_config

            value.class_config.config["episode_parameter_provider"] = {"type": self.config.epp_wrapper_path, "config": wrapper_config}

        return config

    def get_num_test_cases(self) -> int:
        return self.config.num_test_cases

    def get_test_cases(self):
        """
        Method responsible for retrieving test cases planned for evaluation.
        Using default strategy does not plan test cases by a collection of parameter values.
        """
        return self.test_cases

    def get_test_case_index(self, episode_id: int) -> TestCaseIndex:  # noqa: PLR6301
        return TestCaseIndex(episode_id)
