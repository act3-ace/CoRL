# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
import numpy as np
import tree

from corl.libraries.units import Quantity


def assert_instance(in_object, desired_type):
    assert isinstance(in_object, desired_type)


def assert_equal_quantity(object1: Quantity, object2: Quantity):
    if isinstance(object1.m, np.ndarray):
        assert np.isclose(object1.m, object2.m).all()
    else:
        assert object1.m == object2.m


def check_observation_glue(created_glue, expected_observation_space, expected_observation):
    assert expected_observation_space == created_glue.observation_space
    assert None is created_glue.action_space
    assert None is created_glue.apply_action({}, {}, {}, {}, {})

    observation = created_glue.get_observation({}, {}, {})
    assert created_glue.observation_prop.contains(observation)
    tree.map_structure(lambda x: assert_instance(x, Quantity), observation)
    tree.map_structure(assert_equal_quantity, observation, expected_observation)
