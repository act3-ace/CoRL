"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import gym
import numpy as np
import pytest
from gym import spaces
import typing

from corl.libraries.property import BoxProp, DiscreteProp, MultiBinary, RepeatedProp


def test_box_prop():

    class TestProp(BoxProp):
        name: str = "test"
        low: typing.List[float] = [0.0]
        high: typing.List[float] = [100.0]
        unit: typing.List[str] = ["none"]
        description: str = "test"

    expected_gym_space = spaces.Box(low = np.array([0.0]), high = np.array([100.0]))
    act3_prop = TestProp()
    output_gym_space = act3_prop.create_space()
    assert isinstance(output_gym_space, spaces.Box)
    assert output_gym_space.low == expected_gym_space.low
    assert output_gym_space.high == expected_gym_space.high
    assert output_gym_space.dtype == expected_gym_space.dtype


def test_multibox_prop():

    class TestProp(BoxProp):
        name: str = "test"
        low: typing.List[float] = [0.0, 1.0]
        high: typing.List[float] = [100.0, 101.0]
        unit: typing.List[str] = ["none", "none"]
        description: str = "test"

    expected_gym_space = spaces.Box(low = np.array([0.0, 1.0]), high=np.array([100.0, 101.0]))
    act3_prop = TestProp()
    output_gym_space = act3_prop.create_space()
    assert isinstance(output_gym_space, spaces.Box)
    assert (output_gym_space.low == expected_gym_space.low).all()
    assert (output_gym_space.high == expected_gym_space.high).all()
    assert output_gym_space.dtype == expected_gym_space.dtype

def test_2d_prop():

    class TestProp(BoxProp):
        name: str = "test"
        low: typing.List[typing.List[int]] = [[0, 0], [0, 0]]
        high: typing.List[typing.List[int]] = [[100, 101], [101, 100]]
        unit: typing.List[typing.List[str]] = [["none", "none"], ["none", "none"]]
        description: str = "test"

    expected_gym_space = spaces.Box(low=np.array([[0, 0], [0, 0]]), high=np.array([[100, 101], [101, 100]]))
    act3_prop = TestProp()
    output_gym_space = act3_prop.create_space()
    assert isinstance(output_gym_space, spaces.Box)
    assert (output_gym_space.low == expected_gym_space.low).all()
    assert (output_gym_space.high == expected_gym_space.high).all()
    assert output_gym_space.dtype == expected_gym_space.dtype

def test_discrete_prop():

    class TestProp(DiscreteProp):
        name: str = "test"
        n: int = 5
        unit: str = "none"
        description: str = "test"

    expected_gym_space = spaces.Discrete(n = 5)
    act3_prop = TestProp()
    output_gym_space = act3_prop.create_space()
    assert isinstance(output_gym_space, spaces.Discrete)
    assert (output_gym_space.n == expected_gym_space.n)

def test_MultiBinary_prop():

    class TestProp(MultiBinary):
        name: str = "test"
        n: int = 5
        unit: str = "none"
        description: str = "test"

    expected_gym_space = spaces.MultiBinary(n = 5)
    act3_prop = TestProp()
    output_gym_space = act3_prop.create_space()
    # until ray2 is fixed for handling multibinary correctly this
    # we are using multidiscrete as a standin
    assert isinstance(output_gym_space, spaces.MultiDiscrete)
    # assert (output_gym_space.n == expected_gym_space.n)
    assert expected_gym_space.contains(output_gym_space.sample())
