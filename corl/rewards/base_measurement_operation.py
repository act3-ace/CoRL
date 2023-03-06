"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Module with base implimentations for Observations
"""
from __future__ import annotations

import logging
import typing

import numpy as np
from pydantic import BaseModel

from corl.libraries.observation_extractor import ExtractorSet, ObservationExtractor, ObservationSpaceExtractor
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator


class ObservationExtractorValidator(BaseModel):
    """
    fields: The Fields the extractor must access to the nested obs
        ex: [ObserveSensor_Sensor_AltitudeDiff, direct_observation_diff]
    indices: List of indices to extract from the glue
    """
    fields: typing.List[typing.Union[int, str]]
    indices: typing.Union[int, typing.List[int]] = []

    def construct_extractors(self):
        """
        Builds extractor methods for extracting value, observation_space, and unit from an observation glue

        Parameters
        ----------
        platforms : List[str]
            The platforms the glue is observing, needed to compute the glue's prefix

        Returns
        -------
        ExtractorSet
            Named Tuple of value, space, and unit extractors
        """

        def obs_extractor(obs, *_, full_extraction=False):
            indices = []
            if full_extraction:
                indices = self.indices
            return ObservationExtractor(observation=obs, fields=self.fields, indices=indices)

        def obs_space_extractor(obs, *_):
            return ObservationSpaceExtractor(observation_space=obs, fields=self.fields)

        def unit_extractor(obs, *_):
            return ObservationSpaceExtractor(observation_space=obs, fields=self.fields)

        return ExtractorSet(obs_extractor, obs_space_extractor, unit_extractor)

    @staticmethod
    def get_curr_and_next_observation(extractor, observation, next_observation, allow_array: bool = False):
        """Helper function to extract the current and next observation
        """
        curr_metric = extractor(observation)
        next_metric = extractor(next_observation)

        # MTB - It seems that rewardDict must be set to a scalar.
        #       Therefore we must extract the scalar value from observation
        #       Unsure how this will work if an observation isn't a ndarray of size 1,
        #       so just making it fail if that ever happens (i.e. kick the can)
        if isinstance(curr_metric, np.ndarray):
            if len(curr_metric) != len(next_metric):
                raise RuntimeError("Length of arrays do not match, this is a nonop")
            if not allow_array:
                if len(curr_metric) != 1:
                    raise RuntimeError("The observation attempting to do potential based shaping is not a scalar, unsure how to procede")
                curr_metric = curr_metric[0]
                next_metric = next_metric[0]
        else:
            raise RuntimeError("The extracted observation is not a type that is known how to handle")

        return (curr_metric, next_metric)


class BaseMeasurementOperationValidator(RewardFuncBaseValidator):
    """
    observation: Dict of observation extractor arguments described in ObservationExtractorValidator
    """
    observation: ObservationExtractorValidator


class BaseMeasurementOperation(RewardFuncBase):  # pylint: disable=abstract-method
    """Base class for any reward that is to operate on a measurement of some kind
    """

    @property
    def get_validator(self) -> typing.Type[BaseMeasurementOperationValidator]:
        return BaseMeasurementOperationValidator

    def __init__(self, **kwargs) -> None:
        self.config: BaseMeasurementOperationValidator
        super().__init__(**kwargs)
        self._logger = logging.getLogger(self.name)
        self.extractor: ExtractorSet = self.config.observation.construct_extractors()
