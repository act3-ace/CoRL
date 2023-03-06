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
import typing

from corl.rewards.base_measurement_operation import ExtractorSet, ObservationExtractorValidator
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator


class MultiMeasurementOperationValidator(RewardFuncBaseValidator):
    """
    observations: Dict of dicts of observation extractor arguments described in ObservationExtractorValidator
    """
    observations: typing.Dict[str, ObservationExtractorValidator]


class MultiMeasurementOperation(RewardFuncBase):  # pylint: disable=abstract-method
    """Base class for any reward that is to operate on multiple measurements of some kind
    """

    @property
    def get_validator(self) -> typing.Type[MultiMeasurementOperationValidator]:
        return MultiMeasurementOperationValidator

    def __init__(self, **kwargs) -> None:
        self.config: MultiMeasurementOperationValidator
        super().__init__(**kwargs)
        self._logger = logging.getLogger(self.name)
        # construct extractors
        self.extractors: typing.Dict[str, ExtractorSet] = {}
        for key, observation in self.config.observations.items():
            self.extractors[key] = observation.construct_extractors()
