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
import typing

from pydantic import BaseModel, PyObject


class TestCaseStrategyValidator(BaseModel):
    """
    Validator for TestCaseStrategy classes.
    """
    config: typing.Dict = {}


class TestCaseStrategy(abc.ABC):
    """
    Abstract class to define template for TestCaseStrategy.
    This interface handles EPP specific test case management and rllib config mutation.
    """

    def __init__(self, **kwargs) -> None:
        self.config: TestCaseStrategyValidator = self.get_validator(**kwargs)

    @property
    def get_validator(self) -> typing.Type[TestCaseStrategyValidator]:
        """
        Method to return validator for TestCaseStrategy
        """
        return TestCaseStrategyValidator

    @abc.abstractmethod
    def update_rllib_config(self, config: dict) -> dict:
        """
        This method is reponsible for mutating the RLLibConfig, overriding or wrapping the EPPs defined in the task configs.

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
    def get_test_cases(self):
        """
        This method is reponsible for fetching the data structure representing configured/scheduled test cases, if available.
        Ex. The TabularParameterProvider's test_cases Pandas.DataFrame

        Returns
        -------
        test_cases
            The evaluation's test cases
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_test_cases(self) -> int:
        """
        This method is reponsible for returning the number of test cases configured.

        Returns
        -------
        num_test_cases
            The number of evaluation episodes to run (ie. the number of test cases)
        """
        raise NotImplementedError


class TestCaseManagerValidator(BaseModel):
    """
    Validator for the TestCaseManager class

    config: typing.Dict
        config for the TestCaseStrategy class constructor
    class_path: PyObject
        desired TestCaseStrategy path
    """
    config: typing.Dict = {}
    class_path: typing.Optional[PyObject] = None


class TestCaseManager:
    """
    Instantiates the desired EPP with config values defined in launch config.
    Handles test cases and EPP configs. Prepares EPP configs.

    Conrete context class to provide stable API for managing test cases and mutating the rllib config appropriately.
    This class delegates most functions to its TestCaseStrategy object, which encapsulates EPP specific logic.
    """

    def __init__(self, class_path: typing.Optional[str] = None, config: typing.Optional[typing.Dict[str, typing.Any]] = None, **kwargs):
        # create strategy instance
        self.config: typing.Optional[TestCaseManagerValidator] = None
        self.test_case_strategy: typing.Optional[typing.Any] = None

        if class_path:
            if config is None:
                config = {}
            self.config = self.get_validator(class_path=class_path, config=config, **kwargs)
            self.test_case_strategy = self.config.class_path(config=self.config.config)  # type: ignore  # pylint: disable=not-callable

    @property
    def get_validator(self) -> typing.Type[TestCaseManagerValidator]:
        """
        Method to return validator for TestCaseManager
        """
        return TestCaseManagerValidator

    # EPP Specific Methods (strategy delegation)
    def update_rllib_config(self, config: dict):
        """
        This method delegates to an EPP-specific strategy to mutate the rllib_config, overwriting a task's EPPs as needed.
        """
        if self.test_case_strategy:
            return self.test_case_strategy.update_rllib_config(config)
        return config

    def get_test_cases(self):
        """
        This method delegates to an EPP-specific strategy to return a collection of test cases set to run during evaluation.
        """
        if self.test_case_strategy:
            return self.test_case_strategy.get_test_cases()
        return None

    def get_num_test_cases(self) -> int:
        """
        This method delegates to an EPP-specific strategy to to return the number of test cases set to run during evaluation.
        """
        if self.test_case_strategy:
            return self.test_case_strategy.get_num_test_cases()
        return 0
