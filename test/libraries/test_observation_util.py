# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

from collections import OrderedDict

from corl.libraries.observation_util import filter_observations, mutate_observations


def build_sample_observations() -> OrderedDict:
    agent_ids = ["red0", "blue0"]
    obs_samples = [
        ("Test_Observation_a", 0.0),
        ("Test_Observation_b", 1.0),
        ("Test_Observation_c", 2.0),
        ("Test_Observation_d", 3.0),
        ("Test_Observation_e", 4.0),
        ("Test_Observation_f", 5.0),
        ("Test_Observation_g", 6.0),
        ("Test_Observation_h", 7.0),
        ("Test_Observation_i", 8.0),
        ("Test_Observation_j", 9.0),
    ]

    observations: OrderedDict = OrderedDict()
    for agent_id in agent_ids:
        observations[agent_id] = OrderedDict()
        for obs_name, obs_value in obs_samples:
            observations[agent_id][obs_name] = obs_value

    return observations


def test_mutate_observations():
    observations = build_sample_observations()

    mutated_observations = mutate_observations(
        observations, lambda agent_id, _obs_name, obs_value: obs_value * 2 if agent_id == "blue0" else 0.0
    )

    for obs_value in mutated_observations["red0"].values():
        assert obs_value == 0.0

    for obs_name, obs_value in mutated_observations["blue0"].items():
        assert observations["blue0"][obs_name] * 2.0 == obs_value


def test_filter_observations():
    observations = build_sample_observations()
    filtered_observations = filter_observations(observations, lambda _agent_id, _obs_name, obs_value: obs_value >= 5.0)

    for obs_samples in filtered_observations.values():
        for obs_value in obs_samples.values():
            assert obs_value >= 5.0
