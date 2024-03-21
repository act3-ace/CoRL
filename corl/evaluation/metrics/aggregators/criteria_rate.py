"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from corl.evaluation.metrics.generator import MetricGeneratorAggregator
from corl.evaluation.metrics.metric import Metric
from corl.evaluation.metrics.types.terminals.rate import Rate
from corl.evaluation.util.condition import Condition


class CriteriaRate(MetricGeneratorAggregator):
    """Aggregate a metric by determining the rate which a criteria is established"""

    condition: Condition

    def __init__(self, **data):
        super().__init__(**data)
        self.condition = Condition(**data["condition"])

    def generate_metric(self, params: list[Metric] | Metric, **kwargs) -> Metric:
        """Generate the metric

        Arguments:
            metrics {typing.Union[typing.List[Metric], Metric]} -- Already computed metrics

        Returns:
            Metric -- Computed Metric
        """

        if isinstance(params, Metric):
            raise RuntimeError("Metrics given to Criteria Rate must be a list (that is an aggregation of lower scope)")

        occurrence = sum(bool(self.condition.func(item)) for item in params)
        return Rate(occurrence, len(params))
