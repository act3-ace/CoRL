"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import pytest

from corl.test_utils import full_training
from corl.train_rl import build_experiment, parse_corl_args

ppo_rllib_config = {
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


# Adjustments to the configuration files used in this test to match the current baseline is authorized, provided you make a post on
# MatterMost -ML-agents with notifications to joblackburn and bheiner that the change was made.
# All other changes, such as commenting out tests or disabling inference or evaluation requires coordination on the -ML-agents
# channel of MatterMost with notifications to joblackburn and bheiner.
# This should always be committed as False; however, if you need to debug this unit test, set it temporarily to True
# @pytest.mark.ray_debug_mode(False)
@pytest.mark.parametrize(
    "experiment_config, test_rllib_config, train_agent",
    [
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1.yml", ppo_rllib_config, False, id="cartpole-v1"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_repeated_obs.yml", ppo_rllib_config, False, id="cartpole-v1-repeated"),
        pytest.param(
            "config/tasks/gymnasium/experiments/cartpole_v1_with_wrapper.yml", ppo_rllib_config, False, id="cartpole-v1-with_wrapper"
        ),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_random.yml", ppo_rllib_config, False, id="cartpole-v1-random"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_scripted.yml", ppo_rllib_config, True, id="cartpole-v1-scripted"),
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_noop_ctrl.yml", ppo_rllib_config, False, id="cartpole-v1-noop_ctrl"),
        pytest.param(
            "config/tasks/gymnasium/experiments/cartpole_v1_dict_wrapper.yml", ppo_rllib_config, True, id="cartpole-v1-dict_wrapper"
        ),
        # pytest.param("config/tasks/docking_1d/experiments/docking_1d.yml", ppo_rllib_config, True,  id="docking-1d"),
        pytest.param("config/tasks/gymnasium/experiments/pendulum_v1.yml", ppo_rllib_config, False, id="pendulum-v1"),
        # pytest.param("config/tasks/pong/experiments/pong.yml", ppo_rllib_config, False,  id="pong ppo"),
        pytest.param("config/tasks/pong/experiments/pong_commander.yml", ppo_rllib_config, True, id="pong commander ppo"),
        pytest.param("config/tasks/pong/experiments/pong_sac.yml", sac_rllib_config, True, id="pong sac"),
    ],
)
def test_tasks(
    experiment_config,
    test_rllib_config,
    train_agent,
    tmp_path,
    self_managed_ray,
):
    args = parse_corl_args(["--cfg", experiment_config])
    experiment_class, experiment_file_validated = build_experiment(args)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, test_rllib_config, tmp_path)

    if train_agent:
        experiment_class.run_experiment(experiment_file_validated)

        # Determine filename of the checkpoint
        checkpoint_glob = list(tmp_path.glob("training/**/checkpoint_000000"))
        assert len(checkpoint_glob) == 1
        checkpoint_glob[0]
