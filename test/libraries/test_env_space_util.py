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

from corl.libraries.env_space_util import EnvSpaceUtil

gym_default_observation_space = spaces.Dict(
    {
        "sensors":
        spaces.Dict(
            {
                "position": spaces.Box(low=-100, high=100, shape=(3, )),
                "velocity": spaces.Box(low=np.array([-1] * 3), high=np.array([1] * 3)),
                "front_cam": spaces.Tuple((
                    spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                    spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                )),
                "rear_cam": spaces.Box(low=0, high=1, shape=(10, 10, 3)),
            }
        ),
        "ext_controller":
        spaces.MultiDiscrete((5, 2, 2)),
        "inner_state":
        spaces.Dict(
            {
                "charge": spaces.Discrete(100),
                "system_checks": spaces.MultiBinary(10),
                "job_status": spaces.Dict({
                    "task": spaces.Discrete(5),
                    "progress": spaces.Box(low=0, high=100, shape=()),
                }),
            }
        ),
    }
)

gym_space_dict_of_boxes = gym.spaces.Dict(
    {
        tuple([
            "control1",
        ]): gym.spaces.Box(low=1, high=10, shape=(1, )),
        tuple([
            "control2",
        ]): gym.spaces.Box(low=np.array([1]), high=np.array([10])),
        tuple([
            "control3",
        ]): gym.spaces.Box(low=1, high=10, shape=()),
        tuple([
            "control4",
            "control5",
        ]): gym.spaces.Box(low=np.array([1, -10]), high=np.array([10, 1])),
    }
)


def test_observation_pass_fail():
    """
    copied the example from gym.spaces and made sure that works. then checked if observation is messed with it throws error
    """
    observation_space = gym_default_observation_space
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
            "sensors":
            spaces.Dict(
                {
                    "front_cam": spaces.Tuple((
                        spaces.Box(low=0, high=1, shape=(3, )),
                        spaces.Box(low=0, high=1, shape=(3, )),
                    )),
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
    observation_space = spaces.Dict({
        "sensors": spaces.Dict({
            "rear_cam": spaces.Box(low=0, high=1, shape=(3, )),
        }),
    })
    observation = observation_space.sample()
    # mess with the space and assert it throws error
    observation["sensors"]["rear_cam"] = [0.1, 0.2, 0.3]
    EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)


def test_scale_space():
    observation_space = gym_default_observation_space
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
    observation_space = gym_default_observation_space
    zero_mean_space = EnvSpaceUtil.zero_mean_space(observation_space)
    test_mean = (zero_mean_space["sensors"]["position"].low + zero_mean_space["sensors"]["position"].high) / 2.0
    assert np.allclose(np.zeros(shape=zero_mean_space["sensors"]["position"].shape), test_mean)

    test_mean = (zero_mean_space["sensors"]["velocity"].low + zero_mean_space["sensors"]["velocity"].high) / 2.0
    assert np.allclose(np.zeros(shape=zero_mean_space["sensors"]["velocity"].shape), test_mean)


def test_normalize_space():
    observation_space = gym_default_observation_space
    out_max = 0.5
    out_min = -0.5
    scaled_space = EnvSpaceUtil.normalize_space(observation_space, out_max=out_max, out_min=out_min)

    assert isinstance(scaled_space, gym.spaces.Space)
    assert np.allclose(scaled_space["sensors"]["position"].low, np.array([out_min] * 3))


def test_get_zero_sample():
    observation_space = gym_default_observation_space
    zero_space = EnvSpaceUtil.get_zero_sample_from_space(observation_space)
    assert np.allclose(
        np.zeros(observation_space["sensors"]["position"].shape),
        zero_space["sensors"]["position"],
    )


def test_get_mean_sample():
    observation_space = gym_default_observation_space
    mean_space_sample = EnvSpaceUtil.get_mean_sample_from_space(observation_space)
    test_mean = (observation_space["sensors"]["position"].low + observation_space["sensors"]["position"].high) / 2.0
    assert np.allclose(test_mean, mean_space_sample["sensors"]["position"])


def test_add_sample():
    observation_space = gym_default_observation_space
    obs1 = observation_space.sample()
    obs2 = observation_space.sample()
    obsadd = EnvSpaceUtil.add_space_samples(observation_space, obs1, obs2)
    assert np.allclose(
        obsadd["sensors"]["position"],
        obs1["sensors"]["position"] + obs2["sensors"]["position"],
    )


def test_clip_space_sample():
    observation_space = gym_default_observation_space
    obs1 = observation_space.sample()
    obs2 = observation_space.sample()
    obsadd = EnvSpaceUtil.add_space_samples(observation_space, obs1, obs2)
    for _ in range(50):
        obsadd = EnvSpaceUtil.add_space_samples(observation_space, obsadd, obs2)
    clipped_ins = EnvSpaceUtil.clip_space_sample_to_space(obsadd, observation_space)
    assert observation_space.contains(clipped_ins)
    assert np.all(np.less_equal(
        clipped_ins["sensors"]["position"],
        observation_space["sensors"]["position"].high,
    ))
    assert np.all(np.greater_equal(
        clipped_ins["sensors"]["position"],
        observation_space["sensors"]["position"].low,
    ))


def test_turn_box_into_tuple_of_discretes():
    observation_space = gym_default_observation_space
    num_actions = 11
    discrete_only_space = EnvSpaceUtil.turn_box_into_tuple_of_discretes(observation_space, num_actions=num_actions)
    assert isinstance(discrete_only_space["sensors"]["position"], gym.spaces.Tuple)

    assert np.allclose(len(discrete_only_space["sensors"]["position"]), 3)

    for i in discrete_only_space["sensors"]["position"].spaces:
        assert np.allclose(i.n, np.array([num_actions]))

    # if isinstance(
    #     discrete_only_space["inner_state"]["job_status"]["progress"],
    #     gym.spaces.MultiDiscrete,
    # ):
    #     assert np.allclose(
    #         discrete_only_space["inner_state"]["job_status"]["progress"].nvec,
    #         np.array([num_actions]),
    #     )


def test_turn_box_into_tuple_of_discretes_but_really_discrete():

    observation_space = gym_space_dict_of_boxes
    num_actions = 11
    discrete_only_space = EnvSpaceUtil.turn_box_into_tuple_of_discretes(observation_space, num_actions=num_actions)
    for key, dos in discrete_only_space.spaces.items():
        for space in dos:
            assert isinstance(space, gym.spaces.Discrete)
            assert space.n == num_actions

    observation_space = gym_space_dict_of_boxes
    num_actions = {
        tuple([
            "control1",
        ]): (11, ),
        tuple([
            "control2",
        ]): (17, ),
        tuple([
            "control3",
        ]): (13, ),
        tuple([
            "control4",
            "control5",
        ]): (9, 7)
    }
    discrete_only_space = EnvSpaceUtil.turn_box_into_tuple_of_discretes(observation_space, num_actions=num_actions)
    for key, dos in discrete_only_space.spaces.items():
        for idx, space in enumerate(dos):
            assert isinstance(space, gym.spaces.Discrete)
            assert space.n == num_actions[key][idx]


def test_turn_discrete_action_back_to_cont():
    observation_space = gym.spaces.Box(low=np.asarray([1, 1, 1, 1]), high=np.asarray([10, 10, 10, 10]))
    num_actions = 11
    discrete_only_space = EnvSpaceUtil.turn_box_into_tuple_of_discretes(observation_space, num_actions=num_actions)

    space_sample = discrete_only_space.sample()
    cont_space = EnvSpaceUtil.turn_discrete_action_back_to_cont(
        original_space=observation_space,
        discrete_only_space=discrete_only_space,
        space_sample=space_sample,
    )

    # EnvSpaceUtil.deep_sanity_check_space_sample(observation_space, cont_space)
    # assert observation_space.contains(cont_space)
    # assert isinstance(cont_space["sensors"]["position"], np.ndarray)
    # assert (cont_space["sensors"]["position"].shape == observation_space["sensors"]["position"].shape)


def test_turn_discrete_action_back_to_cont_but_really_discrete():
    observation_space = gym_default_observation_space
    num_actions = 11
    discrete_only_space = EnvSpaceUtil.turn_box_into_tuple_of_discretes(observation_space, num_actions=num_actions)

    space_sample = discrete_only_space.sample()
    cont_space = EnvSpaceUtil.turn_discrete_action_back_to_cont(
        original_space=observation_space,
        discrete_only_space=discrete_only_space,
        space_sample=space_sample,
    )

    # EnvSpaceUtil.deep_sanity_check_space_sample(observation_space, cont_space)
    # assert observation_space.contains(cont_space)
    # assert isinstance(cont_space[0], np.ndarray)
    # assert cont_space[0].shape[0] == 1


def test_scale_sample_from_space():
    observation_space = gym_default_observation_space
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
    observation_space = gym_default_observation_space
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

    assert np.all(np.less_equal(
        unscaled_sample["sensors"]["position"],
        observation_space["sensors"]["position"].high,
    ))
    assert np.all(np.greater_equal(
        unscaled_sample["sensors"]["position"],
        observation_space["sensors"]["position"].low,
    ))


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
        np.ones(shape=(2, )),
        # mess with the space bounds and assert it throws an error
        np.ones(shape=(3, )) * 5
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
        # Apparently this will pass because gym thinks this is valid... comment until confirm gym does this correctly
        # np.ones(shape=(2,)),
        # mess with the space bounds and assert it throws an error
        np.ones(shape=(3, )) * 5
    ]

    for bad_value in bad_values:
        observation["sensors"]["rear_cam"] = bad_value
        with pytest.raises(ValueError) as exec_info:
            EnvSpaceUtil.deep_sanity_check_space_sample(space=observation_space, sample=observation)
        print(exec_info)


def test_turn_orig_space_box_to_cont_sample_powerspace():
    #######
    # Assure the symmetry check is working
    disc_space = gym.spaces.Discrete(11)  #discrete space
    flag = 0
    try:
        cont_space = gym.spaces.Box(0, 1, shape=[1])  #continuous space
        spaces_likes = (cont_space, disc_space, 1, 2)
        # This should raise a RuntimeError because the cont_space isn't symmetric
        EnvSpaceUtil.turn_orig_space_box_to_cont_sample_powerspace(spaces_likes)
    except:
        flag += 1
    try:
        cont_space = gym.spaces.Box(0, 0, shape=[1])  #continuous space
        spaces_likes = (cont_space, disc_space, 1, 2)
        # This should raise a RuntimeError because the box high and low are same
        EnvSpaceUtil.turn_orig_space_box_to_cont_sample_powerspace(spaces_likes)
    except:
        flag += 1
    try:
        cont_space = gym.spaces.Box(1, -1, shape=[1])  #continuous space
        spaces_likes = (cont_space, disc_space, 1, 2)
        # This should raise a RuntimeError because the box high is less than low
        EnvSpaceUtil.turn_orig_space_box_to_cont_sample_powerspace(spaces_likes)
    except:
        flag += 1
    assert flag == 3

    #######
    # Assure that spaces > 1d work
    multi_disc_space = gym.spaces.Tuple((gym.spaces.Discrete(11), gym.spaces.Discrete(17)))
    cont_space = gym.spaces.Box(np.asarray([-1, -2]), np.asarray([1, 2]), dtype=np.float32)
    spaces_likes = (cont_space, multi_disc_space, (1, 0), (2, 3))
    EnvSpaceUtil.turn_orig_space_box_to_cont_sample_powerspace(spaces_likes)

    multi_disc_space = gym.spaces.Tuple((gym.spaces.Discrete(11), gym.spaces.Discrete(17)))
    cont_space = gym.spaces.Box(-1, 1, shape=(2, ))
    spaces_likes = (cont_space, multi_disc_space, (1, 0), (2, 3))
    EnvSpaceUtil.turn_orig_space_box_to_cont_sample_powerspace(spaces_likes)

    #######
    # Assure this works for even ints, e.g. 2
    n = 11
    power = 2
    disc_space = gym.spaces.Discrete(n)  #discrete space
    cont_space = gym.spaces.Box(-5, 5, shape=[1])  #continuous space
    outactions = []
    for i in range(n):
        spaces_likes = (cont_space, disc_space, i, power)
        outactions.append(EnvSpaceUtil.turn_orig_space_box_to_cont_sample_powerspace(spaces_likes)[0])
    assert np.allclose(
        outactions, [-5.0, -3.1999998, -1.7999998, -0.79999995, -0.19999994, 0.0, 0.20000005, 0.7999998, 1.8000004, 3.1999998, 5.0]
    )
    #######
    # Assure this works for even ints, e.g. 2
    n = 11
    power = 3
    disc_space = gym.spaces.Discrete(n)  #discrete space
    cont_space = gym.spaces.Box(-5, 5, shape=[1])  #continuous space
    outactions = []
    for i in range(n):
        spaces_likes = (cont_space, disc_space, i, power)
        outactions.append(EnvSpaceUtil.turn_orig_space_box_to_cont_sample_powerspace(spaces_likes)[0])
    assert np.allclose(
        outactions, [-5.0, -2.5599997, -1.0799997, -0.31999996, -0.039999977, 0.0, 0.040000018, 0.3199998, 1.0800005, 2.5599997, 5.0]
    )
    #######
    # Assure this works for non-ints, e.g. 1.5
    n = 11
    power = 1.5
    disc_space = gym.spaces.Discrete(n)  #discrete space
    cont_space = gym.spaces.Box(-5, 5, shape=[1])  #continuous space
    outactions = []
    for i in range(n):
        spaces_likes = (cont_space, disc_space, i, power)
        outactions.append(EnvSpaceUtil.turn_orig_space_box_to_cont_sample_powerspace(spaces_likes)[0])
    assert np.allclose(
        outactions,
        [-5.0000005, -3.577709, -2.3237903, -1.2649109, -0.44721362, 0.0, 0.44721392, 1.2649112, 2.3237903, 3.577709, 5.0000005]
    )
