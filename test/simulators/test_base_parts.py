"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from corl.simulators.base_parts import MutuallyExclusiveParts
import pytest


def test_mutually_exclusive_parts():
    part1 = "control_1"
    part2 = "control_2"
    part3 = "control_3"
    part4 = "control_4"
    part5 = "control_5"
    part6 = "control_1"

    exclusive_parts_1234 = MutuallyExclusiveParts({part1, part2, part3, part4})

    assert exclusive_parts_1234.are_parts_mutually_exclusive([part1, part2, part3, part4])
    assert not exclusive_parts_1234.are_parts_mutually_exclusive([part1, part2, part3, part4, part6])
    with pytest.raises(RuntimeError):
        exclusive_parts_1234.are_parts_mutually_exclusive([part1, part2, part3, part4, part5])

    exclusive_parts_1234_plus = MutuallyExclusiveParts({part1, part2, part3, part4}, allow_other_keys=True)
    assert exclusive_parts_1234_plus.are_parts_mutually_exclusive([part1, part2, part3, part4])
    assert not exclusive_parts_1234_plus.are_parts_mutually_exclusive([part1, part2, part3, part4, part6])
    assert exclusive_parts_1234_plus.are_parts_mutually_exclusive([part1, part2, part3, part4, part5])
