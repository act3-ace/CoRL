"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------
"""
import numpy as np
import pytest
import ray

from corl.libraries import parameters
from corl.test_utils.testing_ureg_func import ureg_setup_func


def test_min_step_updater():
    p = parameters.UniformParameter(name="foo", units="foot", low=7, high=12)
    up = parameters.BoundStepUpdater(param=p, name="low", bound=5, step=-1, bound_type="min")

    value = p.config.low

    value = up.do_call(value)
    assert value == 6

    value = up.do_call(value)
    assert value == 5

    value = up.do_call(value)
    assert value == 5

    value = up.do_call(value, reverse=True)
    assert value == 6

    value = up.do_call(value, reverse=True)
    assert value == 7

    value = up.do_call(value, reverse=True)
    assert value == 7


def test_max_step_updater():
    p = parameters.UniformParameter(name="foo", units="foot", low=1, high=3)
    up = parameters.BoundStepUpdater(param=p, name="high", bound=5, step=1, bound_type="max")

    value = p.config.high

    value = up.do_call(value)
    assert value == 4

    value = up.do_call(value)
    assert value == 5

    value = up.do_call(value)
    assert value == 5

    value = up.do_call(value, reverse=True)
    assert value == 4

    value = up.do_call(value, reverse=True)
    assert value == 3

    value = up.do_call(value, reverse=True)
    assert value == 3


def test_constant():
    data = {"type": "corl.libraries.parameters.ConstantParameter", "config": {"name": "altitude", "units": "foot", "value": 10000}}

    rng = np.random.default_rng(seed=0)

    p = parameters.ConstantParameter(**data["config"])
    assert p.get_value(rng, {}).m == data["config"]["value"]
    assert len(p.updaters) == 0

    p2 = parameters.Factory(**data).build()
    assert type(p2) == parameters.ConstantParameter
    assert p2.get_value(rng, {}).m == data["config"]["value"]
    assert len(p2.updaters) == 0


def test_uniform():
    data = {"type": "corl.libraries.parameters.UniformParameter", "config": {"name": "speed", "units": "knot", "low": 240, "high": 260}}

    rng = np.random.default_rng(seed=2)
    rng2 = np.random.default_rng(seed=2)

    p = parameters.UniformParameter(**data["config"])
    assert p.get_value(rng, {}).m == pytest.approx(245.23224268498632)
    assert len(p.updaters) == 0

    p2 = parameters.Factory(**data).build()
    assert type(p2) == parameters.UniformParameter
    assert p2.get_value(rng2, {}).m == pytest.approx(245.23224268498632)
    assert len(p2.updaters) == 0


def test_uniform_updater():
    data = {
        "type": "corl.libraries.parameters.UniformParameter",
        "config": {
            "name": "speed",
            "units": "knot",
            "low": 240,
            "high": 260,
            "update": {
                "low": {"type": "corl.libraries.parameters.BoundStepUpdater", "config": {"bound": 200, "step": -10, "bound_type": "min"}},
                "high": {"type": "corl.libraries.parameters.BoundStepUpdater", "config": {"bound": 300, "step": 10, "bound_type": "max"}},
            },
        },
    }

    rng = np.random.default_rng(seed=2)

    p = parameters.Factory(**data).build()
    assert type(p) == parameters.UniformParameter
    assert p.get_value(rng, {}).m == pytest.approx(245.23224268498632)
    assert len(p.updaters) == 2
    assert "low" in p.updaters
    assert "high" in p.updaters
    assert type(p.updaters["low"]) == parameters.BoundStepUpdater
    assert type(p.updaters["high"]) == parameters.BoundStepUpdater
    p.updaters["low"]()
    p.updaters["high"]()
    assert p.config.low == pytest.approx(data["config"]["low"] + data["config"]["update"]["low"]["config"]["step"])
    assert p.config.high == pytest.approx(data["config"]["high"] + data["config"]["update"]["high"]["config"]["step"])
    assert p.get_value(rng, {}).m == pytest.approx(241.93964573656493)
    for _ in range(5):
        p.updaters["low"](reverse=True)
    assert p.config.low <= p.config.high
    for _ in range(5):
        p.updaters["high"](reverse=True)
    assert p.config.low <= p.config.high


def test_truncated_normal():
    data = {
        "type": "corl.libraries.parameters.TruncatedNormalParameter",
        "config": {"name": "speed", "units": "knot", "mu": 15000, "std": 1000, "half_width_factor": 0.1},
    }

    rng = np.random.default_rng(seed=3)
    rng2 = np.random.default_rng(seed=3)

    p = parameters.TruncatedNormalParameter(**data["config"])
    assert p.get_value(rng, {}).m == pytest.approx(14917.173138115373)
    assert len(p.updaters) == 0

    p2 = parameters.Factory(**data).build()
    assert type(p2) == parameters.TruncatedNormalParameter
    assert p2.get_value(rng2, {}).m == pytest.approx(14917.173138115373)
    assert len(p2.updaters) == 0


def test_truncated_normal_updater():
    data = {
        "type": "corl.libraries.parameters.TruncatedNormalParameter",
        "config": {
            "name": "speed",
            "units": "knot",
            "mu": 15000,
            "std": 1000,
            "half_width_factor": 0.1,
            "update": {
                "std": {"type": "corl.libraries.parameters.BoundStepUpdater", "config": {"bound": 10000, "step": 100, "bound_type": "max"}},
                "half_width_factor": {
                    "type": "corl.libraries.parameters.BoundStepUpdater",
                    "config": {"bound": 0.8, "step": 0.1, "bound_type": "max"},
                },
            },
        },
    }

    rng = np.random.default_rng(seed=3)

    p = parameters.Factory(**data).build()
    assert type(p) == parameters.TruncatedNormalParameter
    assert p.get_value(rng, {}).m == pytest.approx(14917.173138115373)
    assert len(p.updaters) == 2
    assert "std" in p.updaters
    assert "half_width_factor" in p.updaters
    assert type(p.updaters["std"]) == parameters.BoundStepUpdater
    assert type(p.updaters["half_width_factor"]) == parameters.BoundStepUpdater
    p.updaters["std"]()
    p.updaters["half_width_factor"]()
    assert p.config.std == pytest.approx(data["config"]["std"] + data["config"]["update"]["std"]["config"]["step"])
    assert p.config.half_width_factor == pytest.approx(
        data["config"]["half_width_factor"] + data["config"]["update"]["half_width_factor"]["config"]["step"]
    )
    assert p.get_value(rng, {}).m == pytest.approx(14884.753545294836)
    for _ in range(12):
        p.updaters["std"](reverse=True)
    assert p.config.std >= 0
    for _ in range(5):
        p.updaters["half_width_factor"](reverse=True)
    assert p.config.half_width_factor >= 0


def test_choices_numeric():
    data = {
        "type": "corl.libraries.parameters.ChoiceParameter",
        "config": {"name": "altitude", "units": "foot", "choices": [10000, 15000, 20000]},
    }

    rng = np.random.default_rng(seed=0)

    p = parameters.ChoiceParameter(**data["config"])
    assert p.get_value(rng, {}).m == 20000
    assert len(p.updaters) == 0

    p2 = parameters.Factory(**data).build()
    assert type(p2) == parameters.ChoiceParameter
    assert p2.get_value(rng, {}).m == 15000
    assert len(p2.updaters) == 0


def test_choices_string():
    data = {
        "type": "corl.libraries.parameters.ChoiceParameter",
        "config": {"name": "something", "units": "dimensionless", "choices": ["foo", "bar", "baz"]},
    }

    rng = np.random.default_rng(seed=0)

    p = parameters.ChoiceParameter(**data["config"])
    assert p.get_value(rng, {}) == "baz"
    assert len(p.updaters) == 0

    p2 = parameters.Factory(**data).build()
    assert type(p2) == parameters.ChoiceParameter
    assert p2.get_value(rng, {}) == "bar"
    assert len(p2.updaters) == 0


@ray.remote(max_restarts=0, max_task_retries=0)
class RayTestClass:
    def __init__(self, data):
        self.data = data
        self.parameters = [parameters.Factory(**elem).build() for elem in data]
        ureg_setup_func()

    def count(self):
        return len(self.parameters)

    def get(self, i):
        return self.parameters[i]

    def get_value(self, i, rng):
        return self.parameters[i].get_value(rng, {}).m

    def hyperparameters(self, i):
        return self.parameters[i].updaters.keys()

    def update(self, i, hyperparameter):
        method = self.parameters[i].updaters[hyperparameter]
        method()


def test_ray():
    data = [
        {"type": "corl.libraries.parameters.ConstantParameter", "config": {"name": "altitude", "units": "foot", "value": 10000}},
        {"type": "corl.libraries.parameters.UniformParameter", "config": {"name": "speed", "units": "knot", "low": 240, "high": 260}},
        {
            "type": "corl.libraries.parameters.TruncatedNormalParameter",
            "config": {"name": "speed", "units": "knot", "mu": 15000, "std": 1000, "half_width_factor": 0.1},
        },
        {
            "type": "corl.libraries.parameters.UniformParameter",
            "config": {
                "name": "speed",
                "units": "knot",
                "low": 240,
                "high": 260,
                "update": {
                    "low": {
                        "type": "corl.libraries.parameters.BoundStepUpdater",
                        "config": {"bound": 200, "step": -10, "bound_type": "min"},
                    },
                    "high": {
                        "type": "corl.libraries.parameters.BoundStepUpdater",
                        "config": {"bound": 300, "step": 10, "bound_type": "max"},
                    },
                },
            },
        },
        {
            "type": "corl.libraries.parameters.TruncatedNormalParameter",
            "config": {
                "name": "speed",
                "units": "knot",
                "mu": 15000,
                "std": 1000,
                "half_width_factor": 0.1,
                "update": {
                    "half_width_factor": {
                        "type": "corl.libraries.parameters.BoundStepUpdater",
                        "config": {"bound": 0.3, "step": 0.05, "bound_type": "max"},
                    }
                },
            },
        },
    ]

    rng = np.random.default_rng(0)

    handle = RayTestClass.remote(data)
    for i in range(ray.get(handle.count.remote())):
        obj = ray.get(handle.get.remote(i))
        assert isinstance(obj, parameters.Parameter)
        value = ray.get(handle.get_value.remote(i, rng))
        assert isinstance(value, int | float)
        variables = ray.get(handle.hyperparameters.remote(i))
        for var in variables:
            assert isinstance(var, str)
            ray.get(handle.update.remote(i, var))
        value = ray.get(handle.get_value.remote(i, rng))
        assert isinstance(value, int | float)
