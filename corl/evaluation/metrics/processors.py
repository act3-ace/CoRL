"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Contains processors that execute on given data
"""
from __future__ import annotations

import abc
import typing

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.metrics.alerts import Alert, AlertGenerator
from corl.evaluation.metrics.generator import (
    MetricGenerator,
    MetricGeneratorAggregator,
    MetricGeneratorTerminalEvaluationScope,
    MetricGeneratorTerminalEventScope,
)
from corl.evaluation.metrics.metric import Metric, NonTerminalMetric
from corl.evaluation.metrics.scopes import Scopes

Data = typing.TypeVar("Data")


class Processor(abc.ABC):
    """Base class for a metric processor

    A metric processor combines data and generators to generate metrics
    """

    _metrics: dict[str, Metric]
    _alerts: list[Alert]
    _metrics_info: dict[str, str]  # collect metrics information

    def __init__(self, metrics: dict[str, Metric], metrics_info: dict[str, str], alerts: list[Alert] | None = None):
        self._metrics = metrics

        self._alerts = [] if alerts is None else alerts
        self._metrics_info = metrics_info

    @property
    def metrics(self) -> dict[str, Metric]:
        """Returns the calculated metrics"""
        return self._metrics

    @property
    def metrics_info(self) -> dict[str, str]:
        """
        Returns descriptions of all calculated metrics
        """

        return self._metrics_info

    @property
    def alerts(self) -> list[Alert]:
        """Returns the calculated alerts"""
        return self._alerts


class Event(Processor):
    """Process and Event

    An event is a single EpisodeArtifact instance
    """

    _data: EpisodeArtifact

    @property
    def data(self):
        """Get the data on the event"""
        return self._data

    @classmethod
    def process(
        cls, data: EpisodeArtifact, metric_generators: list[MetricGenerator], alert_generators: list[AlertGenerator], **kwargs
    ) -> Event:
        """Generate the metrics and alerts by giving the generators the provided data"""

        metrics_info: dict[str, str] = {}

        ####################
        ## Calculate metrics

        # Calculate any of the terminal metrics on the event
        metrics: dict[str, Metric] = {}
        for generator in metric_generators:
            if isinstance(generator, MetricGeneratorTerminalEventScope):
                if generator.name in metrics:
                    raise RuntimeError(f"Duplicate metric name {generator.name}")
                metrics[generator.name] = generator.generate_metric(data, **kwargs)
                metrics_info[generator.name] = generator.description

        # Calculate any aggregate generators
        for generator in metric_generators:
            if isinstance(generator, MetricGeneratorAggregator):
                aggregator = generator

                # If we have yet to computed the base metric needed for the aggregator then continue
                if aggregator.metrics_to_use not in metrics:
                    continue

                # If the scope is provided and its the wrong kind then skip
                if aggregator.scope is not None and aggregator.scope != Scopes.EVENT:
                    continue

                # If it's not a nonterminal metric then we can't aggregate anything at this point
                if not isinstance(metrics[aggregator.metrics_to_use], NonTerminalMetric):
                    continue

                # Compute and insert
                if generator.name in metrics:
                    raise RuntimeError(f"Duplicate metric name {generator.name}")
                metrics[generator.name] = aggregator.generate_metric(metrics[aggregator.metrics_to_use])
                metrics_info[generator.name] = generator.description

        ####################
        # Calculate alerts
        alerts: list[Alert] = []
        for alert_generator in alert_generators:
            # Determine if we can evaluate the alert
            can_evaluate = True
            if alert_generator.metric not in metrics:
                can_evaluate = False

            # If we can't evaluate alert, but we are marked to be able to, then error otherwise continue
            if can_evaluate is False:
                if alert_generator.scope == Scopes.EVENT:
                    raise RuntimeError(
                        f"Metric of {alert_generator.name} is marked as event scope, \
                        yet can not be called, check metrics are configured correctly"
                    )
                continue

            if isinstance(alert_generator, AlertGenerator):
                alerts += alert_generator.generate(metrics)

        event = Event(metrics, metrics_info, alerts)
        event._data = data  # noqa: SLF001
        return event


class Evaluation(Processor):
    """Process an Evaluation

    An evaluation is a list of events.
    """

    events: list[Event]

    def __init__(self, metrics: dict[str, Metric], metrics_info: dict[str, str], alerts: list[Alert], events: list[Event]):
        super().__init__(metrics, metrics_info, alerts)
        self.events = events

    @classmethod
    def process(
        cls, data: list[EpisodeArtifact], metric_generators: list[MetricGenerator], alert_generators: list[AlertGenerator], **kwargs
    ) -> Evaluation:
        """Generate the metrics by giving the generators the provided data"""

        ####################
        # Process each event with the given generators
        events: list[Event] = [Event.process(item, metric_generators, alert_generators, **kwargs) for item in data]

        ####################
        ## Calculate metrics

        metrics: dict[str, Metric] = {}
        metrics_info: dict[str, str] = {}

        # Calculate any of the terminal metrics on the Evaluation scope
        for generator in metric_generators:
            if isinstance(generator, MetricGeneratorTerminalEvaluationScope):
                if generator.name in metrics:
                    raise RuntimeError(f"Duplicate metric name {generator.name}")
                metrics[generator.name] = generator.generate_metric(data)
                metrics_info[generator.name] = generator.description

        # Calculate any aggregate generators
        for generator in metric_generators:
            if isinstance(generator, MetricGeneratorAggregator):
                aggregator = generator

                # If we have yet to computed the base metric needed for the aggregator then continue
                if aggregator.metrics_to_use not in events[0].metrics:
                    continue

                # If the scope is provided and its the wrong kind then skip
                if aggregator.scope is not None and aggregator.scope != Scopes.EVALUATION:
                    continue

                # If we have already computed the metric on an event then continue
                if aggregator.name in events[0].metrics:
                    continue

                # Extract the desired metric as a list
                metrics_to_give_aggregator = []
                for i, computed_metrics in enumerate(item.metrics for item in events):
                    if aggregator.metrics_to_use not in computed_metrics:
                        raise RuntimeError(f"{aggregator.metrics_to_use} not in the {i}th event of the evaluation")
                    metrics_to_give_aggregator.append(computed_metrics[aggregator.metrics_to_use])

                # Compute and insert
                if generator.name in metrics:
                    raise RuntimeError(f"Duplicate metric name {generator.name}")
                metrics[generator.name] = aggregator.generate_metric(metrics_to_give_aggregator)
                metrics_info[generator.name] = generator.description

        ####################
        # Calculate alerts
        alerts: list[Alert] = []
        for alert_generator in alert_generators:
            # Determine if we can evaluate the alert
            can_evaluate = True
            if alert_generator.metric not in metrics:
                can_evaluate = False

            # If we can't evaluate alert, but we are marked to be able to, then error otherwise continue
            if can_evaluate is False:
                if alert_generator.scope == Scopes.EVALUATION:
                    raise RuntimeError(
                        f"Metric of {alert_generator.name} is marked as event scope, yet can not be called,\
                        check metrics are configured correctly"
                    )
                continue

            if isinstance(alert_generator, AlertGenerator):
                alerts += alert_generator.generate(metrics)

        return Evaluation(metrics, metrics_info, alerts, events)


class Tournament(Processor):
    """Process an Evaluation

    An evaluation is a list of evaluations.
    """

    evaluations: list[Evaluation]

    @classmethod
    def process(
        cls,
        data: list[list[EpisodeArtifact]],
        metric_generators: list[MetricGenerator],
        alert_generators: list[AlertGenerator],
        **kwargs,
    ) -> Processor:
        """Generate the metrics by giving the generators the provided data"""

        raise RuntimeError("Not Tested")
        # # Process each evaluation with the given generators
        # for item in self.evaluations:
        #     item.process(generators)

        # self.metrics: typing.Dict[str, Metric] = {}

        # # Calculate any of the terminal metrics on the Evaluation scope
        # tournament_data_arr = self.data()
        # for generator in generators:
        #     if isinstance(generator, MetricGeneratorTerminalTournamentScope):
        #         if generator.name in self.metrics:
        #             raise RuntimeError(f"Duplicate metric name {generator.name}")
        #         self.metrics[generator.name] = generator.generate_metric(tournament_data_arr)

        # # Calculate any aggregate generators
        # for generator in generators:
        #     if isinstance(generator, MetricGeneratorAggregator):
        #         aggregator = generator

        #         # If we have yet to computed the base metric needed for the aggregator then continue
        #         if aggregator.metrics_to_use not in self.evaluations[0].metrics:
        #             continue

        #         # If the scope is provided and its the wrong kind then skip
        #         if aggregator.scope is not None and aggregator.scope != Scopes.TOURNAMENT:
        #             continue

        #         # Extract the desired metric as a list
        #         metrics_to_give_aggregator = []
        #         for i, metrics in enumerate([item.metrics for item in self.evaluations]):
        #             if aggregator.metrics_to_use not in metrics:
        #                 raise RuntimeError(f"{aggregator.metrics_to_use} not in the {i}th event of the evaluation")
        #             metrics_to_give_aggregator.append(metrics[aggregator.metrics_to_use])

        #         #Compute and insert
        #         if generator.name in self.metrics:
        #             raise RuntimeError(f"Duplicate metric name {generator.name}")
        #         self.metrics[generator.name] = aggregator.generate_metric(metrics_to_give_aggregator)
