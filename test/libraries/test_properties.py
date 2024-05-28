# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

import typing

import numpy as np
import pytest
import tree
from gymnasium import spaces

from corl.libraries.property import BoxProp, CorlRepeated, DictProp, DiscreteProp, MultiBinary, MultiDiscreteProp, Prop, RepeatedProp
from corl.libraries.units import Quantity


class TestBoxProp(BoxProp):
    name: str = "test"
    low: typing.Sequence[float] | typing.Sequence[typing.Sequence[float]] | np.ndarray = [0.0]
    high: typing.Sequence[float] | typing.Sequence[typing.Sequence[float]] | np.ndarray = [100.0]
    unit: str = "dimensionless"
    description: str = "test"

    @staticmethod
    def expected_gymnasium_space():
        return spaces.Box(low=np.array([0.0]), high=np.array([100.0]))

    @staticmethod
    def expected_scaled_prop():
        return TestBoxProp(low=[0.0], high=[20.0])


class TestBoxPropNpArray(BoxProp):
    name: str = "test"
    low: typing.Sequence[float] | typing.Sequence[typing.Sequence[float]] | np.ndarray = np.array([0.0])
    high: typing.Sequence[float] | typing.Sequence[typing.Sequence[float]] | np.ndarray = np.array([100.0])
    unit: str = "dimensionless"
    description: str = "test"

    @staticmethod
    def expected_gymnasium_space():
        return spaces.Box(low=np.array([0.0]), high=np.array([100.0]))

    @staticmethod
    def expected_scaled_prop():
        return TestBoxPropNpArray(low=np.array([0.0]), high=np.array([20.0]))


class TestMultiBoxProp(BoxProp):
    name: str = "test"
    low: typing.Sequence[float] | typing.Sequence[typing.Sequence[float]] | np.ndarray = [0.0, 1.0]
    high: typing.Sequence[float] | typing.Sequence[typing.Sequence[float]] | np.ndarray = [100.0, 101.0]
    unit: str = "dimensionless"
    description: str = "test"

    @staticmethod
    def expected_gymnasium_space():
        return spaces.Box(low=np.array([0.0, 1.0]), high=np.array([100.0, 101.0]))

    @staticmethod
    def expected_scaled_prop():
        return TestMultiBoxProp(low=[0.0, 0.2], high=[20.0, 20.2])


class Test2dBoxProp(BoxProp):
    name: str = "test"
    low: typing.Sequence[float] | typing.Sequence[typing.Sequence[float]] | np.ndarray = [[0, 0], [0, 0]]
    high: typing.Sequence[float] | typing.Sequence[typing.Sequence[float]] | np.ndarray = [[100, 101], [101, 100]]
    unit: str = "dimensionless"
    description: str = "test"

    @staticmethod
    def expected_gymnasium_space():
        return spaces.Box(low=np.array([[0, 0], [0, 0]]), high=np.array([[100, 101], [101, 100]]))

    @staticmethod
    def expected_scaled_prop():
        return Test2dBoxProp(low=[[0.0, 0.0], [0.0, 0.0]], high=[[20.0, 20.2], [20.2, 20.0]])


class TestDiscreteProp(DiscreteProp):
    name: str = "test"
    n: int = 5
    unit: str = "dimensionless"
    description: str = "test"

    @staticmethod
    def expected_gymnasium_space():
        return spaces.Discrete(n=5)

    @staticmethod
    def expected_scaled_prop():
        return TestDiscreteProp()


class TestMultiDiscreteProp(MultiDiscreteProp):
    name: str = "test"
    nvec: np.ndarray | list = [2, 5, 2]
    dtype: np.dtype = np.dtype(np.int64)
    unit: str = "dimensionless"
    description: str = "test"

    def expected_gymnasium_space(self):
        return spaces.MultiDiscrete(self.nvec, dtype=self.dtype)

    @staticmethod
    def expected_scaled_prop():
        return TestMultiDiscreteProp()


class TestMultiBinaryProp(MultiBinary):
    name: str = "test"
    n: int = 5
    unit: str = "dimensionless"
    description: str = "test"

    def expected_gymnasium_space(self):
        return spaces.MultiDiscrete([2] * self.n, dtype=np.int8)  # spaces.MultiBinary(n=5)

    @staticmethod
    def expected_scaled_prop():
        return TestMultiBinaryProp()


class TestDictProp(DictProp):
    spaces: dict[str, Prop] = {
        "field0": TestBoxProp(),
        "field1": TestMultiBoxProp(),
        "field2": Test2dBoxProp(),
        "field3": TestDiscreteProp(),
        "field4": TestMultiBinaryProp(),
        "field5": TestMultiDiscreteProp(),
    }

    def expected_gymnasium_space(self):
        return spaces.Dict({field_name: field_prop.expected_gymnasium_space() for field_name, field_prop in self.items()})

    @staticmethod
    def expected_scaled_prop():
        return TestDictProp(
            spaces={
                "field0": TestBoxProp().expected_scaled_prop(),
                "field1": TestMultiBoxProp().expected_scaled_prop(),
                "field2": Test2dBoxProp().expected_scaled_prop(),
                "field3": TestDiscreteProp().expected_scaled_prop(),
                "field4": TestMultiBinaryProp().expected_scaled_prop(),
                "field5": TestMultiDiscreteProp().expected_scaled_prop(),
            }
        )


class TestRepeatedProp(RepeatedProp):
    max_len: int = 4
    child_space: dict[str, Prop] = {
        "field0": TestBoxProp(),
        "field1": TestMultiBoxProp(),
        "field2": Test2dBoxProp(),
        "field3": TestDiscreteProp(),
        "field4": TestMultiBinaryProp(),
        "field5": TestMultiDiscreteProp(),
    }

    def expected_gymnasium_space(self):
        gym_space = spaces.Dict(
            {
                "field0": self.child_space["field0"].expected_gymnasium_space(),
                "field1": self.child_space["field1"].expected_gymnasium_space(),
                "field2": self.child_space["field2"].expected_gymnasium_space(),
                "field3": self.child_space["field3"].expected_gymnasium_space(),
                "field4": self.child_space["field4"].expected_gymnasium_space(),
                "field5": self.child_space["field5"].expected_gymnasium_space(),
            }
        )
        return CorlRepeated(gym_space, max_len=self.max_len)

    @staticmethod
    def expected_scaled_prop():
        return TestRepeatedProp(
            child_space={
                "field0": TestBoxProp().expected_scaled_prop(),
                "field1": TestMultiBoxProp().expected_scaled_prop(),
                "field2": Test2dBoxProp().expected_scaled_prop(),
                "field3": TestDiscreteProp().expected_scaled_prop(),
                "field4": TestMultiBinaryProp().expected_scaled_prop(),
                "field5": TestMultiDiscreteProp().expected_scaled_prop(),
            }
        )


def assert_instance(in_object, in_type):
    assert isinstance(in_object, in_type)


def assert_equal(object1, object2):
    if isinstance(object1.m, np.ndarray):
        assert (object1.m == object2.m).all()
    else:
        assert object1 == object2


@pytest.mark.parametrize(
    "corl_test_prop",
    [
        pytest.param(TestBoxProp(), id="BoxProp"),
        pytest.param(TestBoxPropNpArray(), id="BoxPropNpArray"),
        pytest.param(TestMultiBoxProp(), id="MultiBoxProp"),
        pytest.param(Test2dBoxProp(), id="2dBoxProp"),
        pytest.param(TestDiscreteProp(), id="DiscreteProp"),
        pytest.param(TestMultiBinaryProp(), id="MultiBinaryProp"),
        pytest.param(TestMultiDiscreteProp(), id="MultiDiscreteProp"),
        pytest.param(TestDictProp(), id="DictProp"),
        pytest.param(TestRepeatedProp(), id="TestRepeatedProp"),
    ],
)
def test_prop(corl_test_prop):
    output_gymnasium_space = corl_test_prop.create_space()
    assert output_gymnasium_space == corl_test_prop.expected_gymnasium_space()

    gym_sample = corl_test_prop.expected_gymnasium_space().sample()
    converted_gym_sample = corl_test_prop.create_quantity(gym_sample)
    np.testing.assert_equal(gym_sample, corl_test_prop.remove_units(converted_gym_sample))

    act3_sample = corl_test_prop.sample()
    converted_act3_sample = corl_test_prop.remove_units(act3_sample)
    tree.map_structure(lambda x: assert_instance(x, Quantity), act3_sample)
    tree.map_structure(assert_equal, act3_sample, corl_test_prop.create_quantity(converted_act3_sample))

    act3_sample = corl_test_prop.sample()
    assert corl_test_prop.contains(act3_sample)
    assert corl_test_prop.expected_gymnasium_space().contains(corl_test_prop.remove_units(act3_sample))

    scaled_prop = corl_test_prop.scale(0.2)
    assert scaled_prop == corl_test_prop.expected_scaled_prop()
