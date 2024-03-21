"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Module with implementation for multiple Observations
"""
import logging

from corl.rewards.base_measurement_operation import ExtractorSet, ObservationExtractorValidator
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator


class MultiMeasurementOperationValidator(RewardFuncBaseValidator):
    """
    observations: Dict of dicts of observation extractor arguments described in ObservationExtractorValidator
    """

    observations: dict[str, ObservationExtractorValidator]


class MultiMeasurementOperation(RewardFuncBase):
    """Base class for any reward that is to operate on multiple measurements of some kind"""

    @staticmethod
    def get_validator() -> type[MultiMeasurementOperationValidator]:
        return MultiMeasurementOperationValidator

    def __init__(self, **kwargs) -> None:
        self.config: MultiMeasurementOperationValidator
        super().__init__(**kwargs)
        self._logger = logging.getLogger(self.name)
        # construct extractors
        self.extractors: dict[str, ExtractorSet] = {}
        for key, observation in self.config.observations.items():
            self.extractors[key] = observation.construct_extractors()
