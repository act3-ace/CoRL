"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import logging
from abc import abstractmethod
from typing import Any, TypeVar

import pandas as pd
from pydantic import field_validator

from corl.episode_parameter_providers.base_network_epp import BaseNetworkParameterProvider, EppUpdate, EppUpdateSignaller
from corl.evaluation.connection.base_eval_connection import BaseEvalConnection
from corl.evaluation.connection.signal import Slot
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import (
    TestCaseIndex,
    TestCaseStrategy,
    TestCaseStrategyValidator,
)
from corl.libraries.context import get_current_context
from corl.libraries.factory import Factory

T = TypeVar("T")


class BaseNetworkTestCaseStrategyValidator(TestCaseStrategyValidator, Factory):
    """
    A validator for the NetworkTestCaseStrategy class
    """

    @field_validator("type", mode="after")
    @classmethod
    def subclass_validator(cls, v, info):
        if not issubclass(v, BaseNetworkParameterProvider):
            raise TypeError(f"Invalid type {v}; must be subclass of {BaseNetworkParameterProvider}")
        return v


class BaseNetworkTestCaseStrategy(TestCaseStrategy, EppUpdateSignaller[T]):
    """
    The concrete implementation of a TestCaseStrategy for the NetworkedParameterProvider.
    """

    def __init__(self, **kwargs) -> None:
        self.config: BaseNetworkTestCaseStrategyValidator
        super().__init__(**kwargs)

        self._logger = logging.getLogger(type(self).__name__)

        self._test_case: pd.DataFrame | list[dict[str, Any]] = [{}]
        self._cur_updates: T | None = None

        connection: BaseEvalConnection | None
        if (connection := get_current_context().get("connection")) is not None:
            connection.modify_epp_signal.register(Slot(self))

    @staticmethod
    def get_validator() -> type[BaseNetworkTestCaseStrategyValidator]:
        """
        Method to return validator for TestCaseStrategy
        """
        return BaseNetworkTestCaseStrategyValidator

    def update_rllib_config(self, config: dict):
        """
        Method responsible for mutating rllib config, overwriting the task's default EPPs with NetworkProvider.
        """
        # Override the environment EPP
        config["env_config"]["episode_parameter_provider"] = {
            "type": f"{self.config.type.__module__}.{self.config.type.__qualname__}",
            "config": {**self.config.config, "registrar": self, "epp_variety": "environment", "agent_id": None},
        }

        # Override the agents EPP
        for agent_id, value in config["env_config"]["agents"].items():
            value.class_config.config["episode_parameter_provider"] = {
                "type": f"{self.config.type.__module__}.{self.config.type.__qualname__}",
                "config": {**self.config.config, "registrar": self, "epp_variety": "agent", "agent_id": agent_id},
            }

        return config

    def get_test_cases(self):
        """
        Method responsible for retrieving test cases planned for evaluation
        """
        return self._test_case

    def get_test_case_index(self, episode_id: int) -> TestCaseIndex:  # noqa: PLR6301
        return TestCaseIndex(episode_id)

    def __call__(self, updates: T):
        self._cur_updates = updates

        self._test_case = self._create_test_case_for_updates(updates)

        self.epp_update_signal(EppUpdate(self.get_num_test_cases() - 1, updates))

    @abstractmethod
    def _create_test_case_for_updates(self, updates: T) -> pd.DataFrame | list[dict[str, Any]]:
        """Adds the most recent test case"""
        ...
