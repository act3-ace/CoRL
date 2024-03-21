"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Classes that represent a metric generator

A metric generator is a class that generates a metric.
"""
import abc
import typing

from pydantic import BaseModel, field_validator

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.metrics.metric import Metric
from corl.evaluation.metrics.scopes import Scopes, from_string

Type = typing.TypeVar("Type")


class MetricGenerator(BaseModel, typing.Generic[Type]):
    """Base class for a metric generator

    A metric generator is a class that is used to create a metric from parameters
    """

    name: str
    """Name of metric
    """
    description: str = "None provided"
    """
    Description of the metric i.e. what is it calculating
    """

    @abc.abstractmethod
    def generate_metric(self, params: Type, **kwargs) -> Metric:
        """Generate the metric from given parameters

        Arguments:
            params {Type} -- Parameters to generate the metric from

        Returns:
            Metric -- Generated metric
        """


class MetricGeneratorTerminalEventScope(abc.ABC, MetricGenerator[EpisodeArtifact]):
    """Base class for a generator that produces metrics from a single Event"""


class MetricGeneratorTerminalEvaluationScope(abc.ABC, MetricGenerator[list[EpisodeArtifact]]):
    """Base class for a generator that produces metrics from an evaluation: a set of events"""


class MetricGeneratorTerminalTournamentScope(abc.ABC, MetricGenerator[list[list[EpisodeArtifact]]]):
    """Base class for a generator that produces metrics from an tournament: a set of evaluations"""


class MetricGeneratorAggregator(abc.ABC, MetricGenerator[Metric | list[Metric]]):
    """Base class for a generator that generates it's metric by aggregating from already generated metrics"""

    metrics_to_use: str
    """Metric to use in aggregation
    """

    scope: Scopes | None = None
    """Scope that the metric may apply to to, set to None to resolve at first available
    """

    @field_validator("scope", mode="before")
    @classmethod
    def _resolve_scope(cls, v):
        return from_string(v) if v is not None and isinstance(v, str) else v
