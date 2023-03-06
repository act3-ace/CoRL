"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
This module defines the TestCaseStrategy for the TabularParameterProvider

Auther: John McCarroll
"""

import typing

import pandas as pd
from pydantic import BaseModel

from corl.evaluation.runners.section_factories.test_cases.pandas import Pandas
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import TestCaseStrategy, TestCaseStrategyValidator


class TabularStrategyValidator(TestCaseStrategyValidator):
    """
    A validator for the TabularStrategy class
    """
    separator: str = "."


class PandasValidator(BaseModel):
    """
    A validator for the Pandas class
    """
    data: typing.Union[str, pd.DataFrame]
    source_form: str
    seed: int = 12345678903141592653589793
    samples: typing.Optional[float] = None
    randomize: bool = True

    class Config:
        """
        config for PandasValidator
        """
        arbitrary_types_allowed = True


class TabularStrategy(TestCaseStrategy):
    """
    The concrete implementation of a TestCaseStrategy for the TabularParameterProvider.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config: TabularStrategyValidator = self.get_validator(**kwargs)
        dataframe_config: PandasValidator = self.get_pandas_validator(**self.config.config)
        self.test_cases = self.generate_pandas_dataframe(dataframe_config)  # type: ignore
        self.epp_class_path = 'corl.episode_parameter_providers.tabular_parameter_provider.TabularParameterProvider'

    @property
    def get_validator(self) -> typing.Type[TabularStrategyValidator]:
        """
        Method to return validator for TestCaseStrategy
        """
        return TabularStrategyValidator

    @property
    def get_pandas_validator(self) -> typing.Type[PandasValidator]:
        """
        Method to return validator for PandasValidator
        """
        return PandasValidator

    def update_rllib_config(self, config: dict):
        """
        Method responsible for mutating rllib config, overwriting the task's default EPPs with TabularParameterProviders.
        """
        # Override the environment EPP

        # retreive epp config parse for environment's params
        epp_env_config = self.parse_config_for_eval()

        config['env_config']['episode_parameter_provider'] = {'type': self.epp_class_path, 'config': epp_env_config}

        # Override the agents EPP
        for agent_id, value in config['env_config']['agents'].items():

            # retreive epp config parsed for agent's params
            epp_agent_config = self.parse_config_for_eval(agent_id=agent_id)

            value.class_config.config['episode_parameter_provider'] = {'type': self.epp_class_path, 'config': epp_agent_config}

        return config

    def parse_config_for_eval(self, agent_id: str = None) -> dict:
        """
        A method responsible for generating the config dict for each TabularParameterProvider
        """

        # isolate subset of params for scope of single epp instance
        subset_dataframe = None

        if agent_id:
            # parse epp config for specified agent's params
            agent_columns = [x for x in self.test_cases.columns if x.startswith(agent_id)]
            agent_test_cases = self.test_cases.loc[:, agent_columns]
            subset_dataframe = agent_test_cases.rename(columns={x: x.replace(agent_id + '.', '') for x in agent_columns})
        else:
            # parse eval config for environment params
            environment_columns = [x for x in self.test_cases.columns if x.startswith('environment')]
            environment_test_cases = self.test_cases.loc[:, environment_columns]
            subset_dataframe = environment_test_cases.rename(columns={x: x.replace('environment.', '') for x in environment_columns})

        # create epp instance config
        epp_config = {'separator': self.config.separator, 'data': subset_dataframe}

        return epp_config

    def get_test_cases(self):
        """
        Method responsible for retreiving test cases planned for evaluation
        """
        return self.test_cases

    def get_num_test_cases(self) -> int:
        """
        Method responsible for reporting the number of test cases planned for evaluation
        """
        return len(self.test_cases.index.values)

    def generate_pandas_dataframe(self, config: dict) -> pd.DataFrame:
        """
        Method responsible for generating the pandas.DataFrame used to maintain test case param values for TabularParameterProvider
        """

        pandas_manager = Pandas(**dict(config))
        pandas_manager.generate()
        dataframe = pandas_manager.data_frame

        return dataframe
