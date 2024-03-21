"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Define the abstract TestCaseManager and TestCaseStrategy classes, which provide the context and strategy for
running evaluation episodes with a variety of EPPs.

Author: John McCarroll
"""

import abc
from typing import Any

import pandas as pd
from pydantic import BaseModel

TestCaseIndex = int


class TestCaseStrategyValidator(BaseModel):
    """
    Validator for TestCaseStrategy classes.
    """

    # config: dict = {}


class TestCaseStrategy(abc.ABC):
    """
    Abstract class to define template for TestCaseStrategy.
    This interface handles EPP specific test case management and rllib config mutation.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config: TestCaseStrategyValidator = self.get_validator()(**kwargs)

    @staticmethod
    def get_validator() -> type[TestCaseStrategyValidator]:
        """
        Method to return validator for TestCaseStrategy
        """
        return TestCaseStrategyValidator

    @abc.abstractmethod
    def update_rllib_config(self, config: dict) -> dict:
        """
        This method is responsible for mutating the RLLibConfig, overriding or wrapping the EPPs defined in the task configs.

        Parameters
        ----------
        config : dict
            The task's rllib config

        Returns
        -------
        Tconfig : dict
            The task's rllib config
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_cases(self) -> pd.DataFrame | list[dict[str, Any]]:
        """
        This method is responsible for fetching the data structure representing configured/scheduled test cases.
        This should NOT return an empty list or empty dataframe. The canonical 'empty' or default test case is [{}]
        which indicates that there are no changes to the default epp parameters.

        Returns
        -------
        test_cases
            The evaluation's test cases
        """
        raise NotImplementedError

    def get_num_test_cases(self) -> int:
        """
        This method is responsible for returning the number of test cases to run. There should be at minimum
        1 test case.
        NOTE: This is not necessarially the same as len(self.test_cases).

        Returns
        -------
        num_test_cases
            The number of evaluation episodes to run (ie. the number of test cases)
        """
        num_test_cases = 0

        test_cases = self.get_test_cases()

        if isinstance(test_cases, pd.DataFrame):
            num_test_cases = len(test_cases.index)

        elif isinstance(test_cases, list):
            num_test_cases = len(test_cases)

        else:
            raise RuntimeError(f"Invalid test_cases {test_cases}\nExpected [pd.DataFrame, list], got type '{type(test_cases)}'")

        if num_test_cases <= 0:
            raise RuntimeError("Too few test cases")

        return num_test_cases

    @abc.abstractmethod
    def get_test_case_index(self, episode_id: int) -> TestCaseIndex:
        """Gets the index of the test case for the specified episode_id"""
        raise NotImplementedError

    def get_test_case(self, index: TestCaseIndex) -> pd.Series | dict[str, Any]:
        return self.get_test_cases()[index]


class NoTestCases(TestCaseStrategy):
    def update_rllib_config(self, config: dict) -> dict:  # noqa: PLR6301
        return config

    def get_test_cases(self) -> pd.DataFrame | list[dict[str, Any]]:  # noqa: PLR6301
        return [{}]

    def get_test_case_index(self, episode_id: int) -> TestCaseIndex:  # noqa: PLR6301
        return 0
