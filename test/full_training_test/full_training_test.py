# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
from pathlib import Path

import pytest

from corl.test_utils import full_training
from corl.train_rl import build_experiment, parse_corl_args

ppo_rllib_config = {
    "horizon": 20,
    "rollout_fragment_length": 10,
    "train_batch_size": 10,
    "sgd_minibatch_size": 10,
    "num_workers": 1,
    "num_cpus_per_worker": 1,
    "num_envs_per_worker": 1,
    "num_cpus_for_driver": 1,
    "num_gpus_per_worker": 0,
    "num_gpus": 0,
    "num_sgd_iter": 5,
    "seed": 1,
}

sac_rllib_config = {
    "horizon": 20,
    "rollout_fragment_length": 10,
    "train_batch_size": 10,
    "num_workers": 1,
    "num_cpus_per_worker": 1,
    "num_envs_per_worker": 1,
    "num_cpus_for_driver": 1,
    "num_gpus_per_worker": 0,
    "num_gpus": 0,
    "seed": 1,
}


# The following tests should be run when upgrading ray/rllib
@pytest.mark.parametrize(
    "experiment_config, test_rllib_config",
    [
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1.yml", ppo_rllib_config, id="cartpole-v1"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_repeated_obs.yml", ppo_rllib_config, id="cartpole-v1-repeated"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_with_wrapper.yml", ppo_rllib_config, id="cartpole-v1-with_wrapper"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_random.yml", ppo_rllib_config, id="cartpole-v1-random"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_scripted.yml", ppo_rllib_config, id="cartpole-v1-scripted"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_noop.yml", ppo_rllib_config, id="cartpole-v1-cartpole_v1_noop"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_noop_ctrl.yml", ppo_rllib_config, id="cartpole-v1-noop_ctrl"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_benchmark.yml", ppo_rllib_config, id="cartpole_v1_benchmark"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_dict_wrapper.yml", ppo_rllib_config, id="cartpole-v1-dict_wrapper"),
        pytest.param("config/tasks/gymnasium/experiments/pendulum_v1.yml", ppo_rllib_config, id="pendulum-v1"),
        pytest.param("config/tasks/docking_1d/experiments/docking_1d.yml", ppo_rllib_config, id="docking-1d"),
        pytest.param("config/tasks/pong/experiments/pong.yml", ppo_rllib_config, id="pong ppo"),
        pytest.param("config/tasks/pong/experiments/pong_alt.yml", ppo_rllib_config, id="pong_alt"),
        pytest.param("config/tasks/pong/experiments/pong_commander.yml", ppo_rllib_config, id="pong commander ppo"),
        pytest.param("config/tasks/pong/experiments/pong_sac.yml", sac_rllib_config, id="pong sac"),
        pytest.param(
            "config/tasks/pong/experiments/pong-one_scripted-random_action.yml", ppo_rllib_config, id="pong-one_scripted-random_action"
        ),
        pytest.param(
            "config/tasks/pong/experiments/pong-both_scripted-random_action.yml", ppo_rllib_config, id="pong-both_scripted-random_action"
        ),
        pytest.param("config/tasks/pong/experiments/pong-one_scripted-noop.yml", ppo_rllib_config, id="pong one_scripted noop.yml"),
        pytest.param("config/tasks/pong/experiments/pong_centralized_critic.yml", ppo_rllib_config, id="pong centrailized critic"),
        pytest.param(
            "config/tasks/pong/experiments/pong-one_scripted-centralized-critic.yml",
            ppo_rllib_config,
            id="pong one_scripted centralized-critic",
        ),
        pytest.param("config/tasks/pong/experiments/pong_group.yml", ppo_rllib_config, id="pong_group"),
    ],
)
@pytest.mark.skip(reason="Full training tests are exercised during ray/rlib updates")
def test_rllib_training(
    experiment_config,
    test_rllib_config,
    tmp_path,
    self_managed_ray,
):
    args = parse_corl_args(["--cfg", experiment_config])
    experiment_class, experiment_file_validated = build_experiment(args)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, test_rllib_config, tmp_path)

    experiment_class.run_experiment(experiment_file_validated)

    # Determine filename of the checkpoint
    checkpoint_glob = list(tmp_path.glob("training/**/checkpoint_000000"))
    assert len(checkpoint_glob) == 1
    checkpoint_glob[0]


@pytest.mark.skip(reason="Full training tests are exercised during ray/rlib updates")
def test_configs_validated(request, all_experiments):
    tested_configs = [Path(test_tasks_args.values[0]) for test_tasks_args in request.module.test_rllib_training.pytestmark[0].args[1]]
    # These experiments are tested elsewhere or are exempt
    exempt_experiments = [
        # tested in offline_training_test
        Path("config/tasks/pong/experiments/pong_marwil.yml")
    ]
    missing_tests = set(all_experiments) - set(tested_configs) - set(exempt_experiments)

    assert len(missing_tests) == 0
