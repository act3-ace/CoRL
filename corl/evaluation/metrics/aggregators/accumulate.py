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
from corl.evaluation.metrics.metric import Metric, TerminalMetric
from corl.evaluation.metrics.types.nonterminals.vector import Vector


class Accumulate(MetricGeneratorAggregator):
    """Aggregate by accumulating.
    """

    def generate_metric(self, params: typing.Union[typing.List[Metric], Metric], **kwargs) -> Metric:
        """Generate the metric

        Arguments:
            params {typing.Union[typing.List[Metric], Metric]} -- Already computed metrics

        Returns:
            Metric -- Computed Metric
        """

        arr: typing.List[Metric]
        if isinstance(params, Metric):
            arr = [params]
        elif isinstance(params, list):
            arr = []
            for idx, item in enumerate(params):
                if not isinstance(item, Metric):
                    raise RuntimeError(f"{idx}th item in list of metrics to accumulate is not of type \"Metric\"")

                # If it's a terminal metric then just merge
                if isinstance(item, TerminalMetric):
                    arr.append(item)
                # If it's a Vector then append
                elif isinstance(item, Vector):
                    for vector_item in item.arr:
                        arr.append(vector_item)
                else:
                    raise RuntimeError(f"Unsure how to merge {type(item)}")
        else:
            raise RuntimeError("Expecting either a singular metric or a list of Metrics")

        return Vector(arr)
