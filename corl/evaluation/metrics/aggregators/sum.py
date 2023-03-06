"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import typing

from corl.evaluation.metrics.generator import MetricGeneratorAggregator
from corl.evaluation.metrics.metric import Metric, NonTerminalMetric
from corl.evaluation.metrics.types.nonterminals.timed_value import TimedValue
from corl.evaluation.metrics.types.nonterminals.vector import Vector
from corl.evaluation.metrics.types.terminals.void import Void


class Sum(MetricGeneratorAggregator):
    """Aggregate by summing.
    """

    def generate_metric(self, params: typing.Union[typing.List[Metric], Metric], **kwargs) -> Metric:
        """Generate the metric

        Arguments:
            metrics {typing.Union[typing.List[Metric], Metric]} -- Already computed metrics

        Returns:
            Metric -- Computed Metric
        """

        arr: typing.List[typing.Any]
        if isinstance(params, NonTerminalMetric):

            if isinstance(params, Vector):
                arr = params.arr
            else:
                raise RuntimeError(f"No known way to compute average of {type(params)}")

        elif isinstance(params, list):
            arr = params
        else:
            RuntimeError("Must either given a NonTerminalMetric or a list of terminals")

        total: typing.Optional[Metric] = None
        for item in arr:

            if isinstance(item, TimedValue):
                item = item.value

            if total is None:
                total = item
            else:
                total = total + item

        # if we haven't set any data then there is an issue
        if total is None:
            return Void()

        return total
