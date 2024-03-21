"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import itertools
import typing
from collections import Counter

from corl.evaluation.metrics.generator import MetricGeneratorAggregator
from corl.evaluation.metrics.metric import Metric
from corl.evaluation.metrics.types.nonterminals.dict import Dict
from corl.evaluation.metrics.types.nonterminals.vector import Vector
from corl.evaluation.metrics.types.terminals.discrete import Discrete
from corl.evaluation.metrics.types.terminals.rate import Rate
from corl.evaluation.metrics.types.terminals.real import Real
from corl.evaluation.metrics.types.terminals.string import String


class TerminalCounterMetric(MetricGeneratorAggregator):
    """Count the instances of each value, returning count or percentage"""

    divide_total: bool = False

    def generate_metric(self, params: list[Metric] | Metric, **kwargs) -> Metric:
        """Generate the metric

        Arguments:
            params {typing.Union[typing.List[Metric], Metric]} -- Already computed metrics

        Returns:
            Metric -- Computed Metric
        """

        count: typing.Counter[typing.Any] = Counter()
        arr: list[Metric] = [params] if isinstance(params, Metric) else params
        for elem in arr:
            count.update(self._get_values(elem))

        if self.divide_total:
            return Dict({k: Real(v / len(arr)) for k, v in count.items()})

        return Dict({k: Rate(v, len(arr)) for k, v in count.items()})

    @classmethod
    def _get_values(cls, metric_elem: Metric) -> typing.Iterable[typing.Any]:
        if isinstance(metric_elem, Real | String | Discrete):
            return [metric_elem.value]

        if isinstance(metric_elem, Rate):
            return [metric_elem.occurrence / metric_elem.total]

        if isinstance(metric_elem, Vector):
            return itertools.chain.from_iterable(cls._get_values(x) for x in metric_elem.arr)

        raise TypeError(f"Unsupported metric: {type(metric_elem).__name__}")
