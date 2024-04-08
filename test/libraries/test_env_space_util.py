"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from collections import OrderedDict

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces

from corl.libraries.env_space_util import EnvSpaceUtil, SingleLayerDict, convert_gymnasium_space, convert_sample
from corl.libraries.units import corl_get_ureg

gymnasium_default_observation_space = spaces.Dict(
    {
        "sensors": spaces.Dict(
            {
                "position": spaces.Box(low=-100, high=100, shape=(3,)),
                "velocity": spaces.Box(low=np.array([-1] * 3), high=np.array([1] * 3)),
                "front_cam": spaces.Tuple(
                    (
                        spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                        spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                    )
                ),
                "rear_cam": spaces.Box(low=0, high=1, shape=(10, 10, 3)),
            }
        ),
        "ext_controller": spaces.MultiDiscrete((5, 2, 2)),
        "inner_state": spaces.Dict(
            {
                "charge": spaces.Discrete(100),
                "system_checks": spaces.MultiBinary(10),
                "job_status": spaces.Dict(
                    {
                        "task": spaces.Discrete(5),
                        "progress": spaces.Box(low=0, high=100, shape=()),
                    }
                ),
            }
        ),
    }
)

gymnasium_space_dict_of_boxes = gymnasium.spaces.Dict(
    {
        ("control1",): gymnasium.spaces.Box(low=1, high=10, shape=(1,)),
        ("control2",): gymnasium.spaces.Box(low=np.array([1]), high=np.array([10])),
        ("control3",): gymnasium.spaces.Box(low=1, high=10, shape=()),
        ("control4", "control5"): gymnasium.spaces.Box(low=np.array([1, -10]), high=np.array([10, 1])),
    }
)


def _check_nested_dict_keys(expected, actual):
    """Recursive function that checks the order of all keys and nested keys of a dict"""
    if isinstance(expected, dict | OrderedDict) and isinstance(actual, dict | OrderedDict):
        keys_expected = list(expected.keys())
        keys_actual = list(actual.keys())
        np.testing.assert_equal(
            actual=keys_actual, desired=keys_expected, err_msg=f"Order of the keys have changed. {keys_expected} != {keys_actual}"
        )

        values_expected = [value for value in expected.values() if isinstance(value, dict | OrderedDict)]
        values_actual = [value for value in actual.values() if isinstance(value, dict | OrderedDict)]
        np.testing.assert_equal(len(values_expected), len(values_actual))
        for nested_dict_expected, nested_dict_actual in zip(values_expected, values_actual):
            _check_nested_dict_keys(nested_dict_expected, nested_dict_actual)


def test_observation_pass_fail():
    """
    copied the example from gymnasium.spaces and made sure that works. then checked if observation is messed with it throws error
    """
    observation_space = gymnasium_default_observation_space
    observation = observation_space.sample()
    EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)

    # mess with the space and assert it throws error
    observation["sensors"]["position"] = np.array([1])
    with pytest.raises(ValueError) as exec_info:
        EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)
    print(f"\n{exec_info}")


def test_observation_type_mismatch():
    """
    If the observation space is a box the observation cannot be a tuple since the tuple can only have more spaces in it
    """
    observation_space = spaces.Dict(
        {
            "sensors": spaces.Dict(
                {
                    "front_cam": spaces.Tuple(
                        (
                            spaces.Box(low=0, high=1, shape=(3,)),
                            spaces.Box(low=0, high=1, shape=(3,)),
                        )
                    ),
                    "rear_cam": spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                }
            ),
        }
    )
    observation = observation_space.sample()
    # mess with the space and assert it throws error
    observation["sensors"]["front_cam"] = ((0, 1, 2), (1, 2, 3))
    with pytest.raises(ValueError) as exec_info:
        EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)
    print(f"\n{exec_info}")


def test_observation_box_as_list():
    """
    We must handle treating boxes as both numpy arrays and lists
    """
    observation_space = spaces.Dict(
        {
            "sensors": spaces.Dict(
                {
                    "rear_cam": spaces.Box(low=0, high=1, shape=(3,)),
                }
            ),
        }
    )
    observation = observation_space.sample()
    # mess with the space and assert it throws error
    observation["sensors"]["rear_cam"] = [0.1, 0.2, 0.3]
    EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)


def test_scale_space():
    observation_space = gymnasium_default_observation_space
    scale: float = 2.0
    scaled_space = EnvSpaceUtil.scale_space(observation_space, scale=scale)
    assert np.allclose(
        observation_space["sensors"]["position"].low * scale,
        scaled_space["sensors"]["position"].low,
    )
    assert np.allclose(
        observation_space["sensors"]["position"].high * scale,
        scaled_space["sensors"]["position"].high,
    )


def test_zero_mean_space_sample():
    observation_space = gymnasium_default_observation_space
    zero_mean_space = EnvSpaceUtil.zero_mean_space(observation_space)
    test_mean = (zero_mean_space["sensors"]["position"].low + zero_mean_space["sensors"]["position"].high) / 2.0
    assert np.allclose(np.zeros(shape=zero_mean_space["sensors"]["position"].shape), test_mean)

    test_mean = (zero_mean_space["sensors"]["velocity"].low + zero_mean_space["sensors"]["velocity"].high) / 2.0
    assert np.allclose(np.zeros(shape=zero_mean_space["sensors"]["velocity"].shape), test_mean)


def test_normalize_space():
    observation_space = gymnasium_default_observation_space
    out_max = 0.5
    out_min = -0.5
    scaled_space = EnvSpaceUtil.normalize_space(observation_space, out_max=out_max, out_min=out_min)

    assert isinstance(scaled_space, gymnasium.spaces.Space)
    assert np.allclose(scaled_space["sensors"]["position"].low, np.array([out_min] * 3))


def test_get_zero_sample():
    observation_space = gymnasium_default_observation_space
    zero_space = EnvSpaceUtil.get_zero_sample_from_space(observation_space)
    assert np.allclose(
        np.zeros(observation_space["sensors"]["position"].shape),
        zero_space["sensors"]["position"],
    )


def test_get_mean_sample():
    observation_space = gymnasium_default_observation_space
    mean_space_sample = EnvSpaceUtil.get_mean_sample_from_space(observation_space)
    test_mean = (observation_space["sensors"]["position"].low + observation_space["sensors"]["position"].high) / 2.0
    assert np.allclose(test_mean, mean_space_sample["sensors"]["position"])


def test_add_sample():
    observation_space = gymnasium_default_observation_space
    obs1 = observation_space.sample()
    obs2 = observation_space.sample()
    obsadd = EnvSpaceUtil.add_space_samples(observation_space, obs1, obs2)
    assert np.allclose(
        obsadd["sensors"]["position"],
        obs1["sensors"]["position"] + obs2["sensors"]["position"],
    )


def test_clip_space_sample():
    observation_space = gymnasium_default_observation_space
    _Quantity = corl_get_ureg().Quantity
    obs1 = observation_space.sample()
    obs2 = observation_space.sample()
    obsadd = EnvSpaceUtil.add_space_samples(observation_space, obs1, obs2)
    for _ in range(50):
        obsadd = EnvSpaceUtil.add_space_samples(observation_space, obsadd, obs2)
    import tree

    obsass_w_units = tree.map_structure(lambda x: _Quantity(x, "dimensionless"), obsadd)
    clipped_ins = EnvSpaceUtil.clip_space_sample_to_space(obsass_w_units, observation_space)
    ununited = tree.map_structure(lambda x: x.m, clipped_ins)
    assert observation_space.contains(ununited)
    assert np.all(
        np.less_equal(
            clipped_ins["sensors"]["position"].m,
            observation_space["sensors"]["position"].high,
        )
    )
    assert np.all(
        np.greater_equal(
            clipped_ins["sensors"]["position"].m,
            observation_space["sensors"]["position"].low,
        )
    )


def test_scale_sample_from_space():
    observation_space = gymnasium_default_observation_space
    obs_sample = observation_space.sample()
    out_max = 0.5
    out_min = -0.5
    obs_sample["sensors"]["position"][0] = observation_space["sensors"]["position"].high[0]
    obs_sample["sensors"]["position"][1] = observation_space["sensors"]["position"].low[1]
    scaled_sample = EnvSpaceUtil.scale_sample_from_space(
        space=observation_space,
        space_sample=obs_sample,
        out_max=out_max,
        out_min=out_min,
    )

    assert np.all(np.less_equal(scaled_sample["sensors"]["position"], out_max))
    assert np.all(np.greater_equal(scaled_sample["sensors"]["position"], out_min))


def test_unscale_sample_from_space():
    observation_space = gymnasium_default_observation_space
    out_max = 0.5
    out_min = -0.5
    scaled_space = EnvSpaceUtil.normalize_space(observation_space, out_max=out_max, out_min=out_min)
    scaled_sample = scaled_space.sample()

    scaled_sample["sensors"]["position"][0] = scaled_space["sensors"]["position"].high[0]
    scaled_sample["sensors"]["position"][1] = scaled_space["sensors"]["position"].low[1]
    unscaled_sample = EnvSpaceUtil.unscale_sample_from_space(
        space=observation_space,
        space_sample=scaled_sample,
        out_max=out_max,
        out_min=out_min,
    )

    assert np.all(
        np.less_equal(
            unscaled_sample["sensors"]["position"],
            observation_space["sensors"]["position"].high,
        )
    )
    assert np.all(
        np.greater_equal(
            unscaled_sample["sensors"]["position"],
            observation_space["sensors"]["position"].low,
        )
    )


def test_deep_merge_dict():
    a = {"first": {"all_rows": {"pass": "dog", "number": "1"}}}
    b = {"first": {"all_rows": {"fail": "cat", "number": "5"}}}
    merged_dict = EnvSpaceUtil.deep_merge_dict(b, a)
    correct_dict = {"first": {"all_rows": {"pass": "dog", "fail": "cat", "number": "5"}}}
    assert merged_dict == correct_dict


def test_observation_discrete_as_np_integer():
    """
    We must handle treating discrete as both numpy integers
    """
    observation_space = spaces.Dict({"sensors": spaces.Dict({"rear_cam": spaces.Discrete(n=3)})})
    observation = observation_space.sample()
    # mess with the space and assert it does not throw an error
    observation["sensors"]["rear_cam"] = np.int64(2)
    EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)


def test_observation_multidiscrete_as_np_integer():
    """
    test multi discrete
    """
    observation_space = spaces.Dict({"sensors": spaces.Dict({"rear_cam": spaces.MultiDiscrete(nvec=[3, 2, 4])})})
    observation = observation_space.sample()
    # the build in sample should pass
    EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)

    bad_values = [
        # mess with the space type and assert it throws an error
        (1, 1, 1),
        # mess with the space shape and assert it throws an error
        np.ones(shape=(2,)),
        # mess with the space bounds and assert it throws an error
        np.ones(shape=(3,)) * 5,
    ]

    for bad_value in bad_values:
        observation["sensors"]["rear_cam"] = bad_value
        with pytest.raises(ValueError) as exec_info:
            EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)
        print(exec_info)


def test_observation_multibinary_as_np_integer():
    """
    test multi discrete
    """
    observation_space = spaces.Dict({"sensors": spaces.Dict({"rear_cam": spaces.MultiBinary(n=3)})})
    observation = observation_space.sample()
    # the build in sample should pass
    EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)

    bad_values = [
        # mess with the space type and assert it throws an error
        (1, 1, 1),
        # mess with the space size and assert it throws an error
        # Apparently this will pass because gymnasium thinks this is valid... comment until confirm gymnasium does this correctly
        # np.ones(shape=(2,)),
        # mess with the space bounds and assert it throws an error
        np.ones(shape=(3,)) * 5,
    ]

    for bad_value in bad_values:
        observation["sensors"]["rear_cam"] = bad_value
        with pytest.raises(ValueError) as exec_info:
            EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)
        print(exec_info)


@pytest.mark.parametrize(
    "input_space, output_type, valid",
    [
        pytest.param(gymnasium_default_observation_space, SingleLayerDict, True, id="MultiNestedDict -> SingleLayerDict"),
        pytest.param(
            spaces.Dict({"parent_key": spaces.Dict({"child_key": spaces.Discrete(3)})}),
            spaces.Discrete,
            True,
            id="NestDictSingleDiscrete -> Discrete",
        ),
        pytest.param(gymnasium_default_observation_space, spaces.Box, True, id="MultiNestedDict -> Box"),
        pytest.param(gymnasium_space_dict_of_boxes, spaces.Box, True, id="DictOfBoxes -> Box"),
        pytest.param(spaces.MultiDiscrete([3, 10]), spaces.Box, True, id="MultiDiscrete -> Box"),
        pytest.param(spaces.Tuple((spaces.Discrete(3), spaces.Discrete(10))), spaces.Box, True, id="Tuple -> Box"),
        pytest.param(spaces.MultiBinary([3, 2]), spaces.Box, True, id="MultiBinary -> Box"),
        pytest.param(
            spaces.Dict({"key_1": spaces.Discrete(3), "key_2": spaces.Discrete(10)}),
            spaces.Discrete,
            False,
            id="DictMultiDiscrete -> Discrete",
        ),
        pytest.param(
            spaces.Dict({"key": spaces.Tuple((spaces.Discrete(3), spaces.Discrete(10)))}),
            spaces.Discrete,
            False,
            id="TupleMultiDiscrete -> Discrete",
        ),
        pytest.param(spaces.Dict({"key": spaces.Box(low=-100, high=100, shape=(3,))}), spaces.Discrete, False, id="Box -> Discrete"),
        pytest.param(spaces.Dict({"key": spaces.MultiDiscrete([3, 2])}), spaces.Discrete, False, id="MultiDiscrete -> Discrete"),
        pytest.param(spaces.Dict({"key": spaces.MultiBinary([3, 2])}), spaces.Discrete, False, id="MultiBinary -> Discrete"),
    ],
)
def test_convert_gymnasium_space_output_type(input_space, output_type, valid):
    """
    Test converting gymnasium spaces
    """
    if valid:
        output = convert_gymnasium_space(input_space=input_space, output_type=output_type)
        assert isinstance(output, output_type)
    else:
        with pytest.raises(ValueError) as exec_info:
            convert_gymnasium_space(input_space=input_space, output_type=output_type)
        print(exec_info)


@pytest.mark.parametrize(
    "input_space, output_type",
    [
        pytest.param(
            spaces.Dict({"parent_key": spaces.Dict({"child_key": spaces.Discrete(10)})}),
            spaces.Discrete,
            id="DictSingleDiscrete -> Discrete",
        ),
        pytest.param(
            spaces.Dict({"parent_key": spaces.Dict({"child_key": spaces.Discrete(10)})}),
            SingleLayerDict,
            id="DictSingleDiscrete -> SingleLayerDict",
        ),
        pytest.param(spaces.Dict({"key": spaces.Box(low=-5, high=5, shape=(3,), dtype=int)}), spaces.Box, id="DictSingleBox -> spaces.Box"),
        pytest.param(
            spaces.Dict(
                {
                    "parent_key": spaces.Dict(
                        {
                            "child_key_2": spaces.Dict(
                                {
                                    "grandchild_key_b": spaces.Discrete(20),
                                    "grandchild_key_c": spaces.Discrete(200),
                                    "grandchild_key_a": spaces.Discrete(2),
                                }
                            ),
                            "child_key_1": spaces.Dict(
                                {
                                    "grandchild_key_bb": spaces.Discrete(20),
                                    "grandchild_key_cc": spaces.Discrete(200),
                                    "grandchild_key_aa": spaces.Discrete(2),
                                }
                            ),
                        }
                    )
                }
            ),
            SingleLayerDict,
            id="DictNestedDiscretes -> SingleLayerDict",
        ),
    ],
)
def test_convert_sample(input_space, output_type):
    """Test converting samples"""
    output_space = convert_gymnasium_space(input_space=input_space, output_type=output_type)

    sample = input_space.sample()

    converted_sample = convert_sample(sample=sample, sample_space=input_space, output_space=output_space)
    assert output_space.contains(converted_sample)

    reverted_sample = convert_sample(sample=converted_sample, sample_space=output_space, output_space=input_space)

    assert input_space.contains(reverted_sample)
    # Fails when the values are not equal.
    # Passes when OrderedDicts are in different order as long as the key and values are the same.
    np.testing.assert_equal(reverted_sample, sample)

    # Will fail if the keys of the dictionaries differ.
    _check_nested_dict_keys(expected=sample, actual=reverted_sample)


@pytest.mark.parametrize(
    "dict_1, dict_2, should_fail",
    [
        pytest.param({3: {4: 5, 1: 2}, 4: 4}, OrderedDict({3: OrderedDict({4: 5, 1: 2}), 4: 4}), False, id="SameOrder-DifferentDictType-1"),
        pytest.param({3: {4: 5, 1: 2}, 4: 4}, OrderedDict({3: {4: 5, 1: 2}, 4: 4}), False, id="SameOrder-DifferentDictType-2"),
        pytest.param({3: {4: 5, 1: 2}, 4: 4}, {3: OrderedDict({4: 5, 1: 2}), 4: 4}, False, id="SameOrder-DifferentDictType-3"),
        pytest.param({3: {4: 5, 1: 2}, 4: 4}, {3: {1: 2, 4: 5}, 4: 4}, True, id="DifferentOrder-PythonDicts"),
        pytest.param(OrderedDict({3: {4: 5, 1: 2}, 4: 4}), OrderedDict({3: {1: 2, 4: 5}, 4: 4}), True, id="DifferentOrder-OrderedDicts"),
    ],
)
def test_check_nested_dict_keys(dict_1, dict_2, should_fail):
    """Test the helper function"""
    if should_fail:
        with pytest.raises(AssertionError) as exec_info:
            _check_nested_dict_keys(dict_1, dict_2)
        print(exec_info)
    else:
        _check_nested_dict_keys(dict_1, dict_2)
