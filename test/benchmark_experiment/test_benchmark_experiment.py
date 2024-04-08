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


# Adjustments to the configuration files used in this test to match the current baseline is authorized, provided you make a post on
# MatterMost -ML-agents with notifications to joblackburn and bheiner that the change was made.
# All other changes, such as commenting out tests or disabling inference or evaluation requires coordination on the -ML-agents
# channel of MatterMost with notifications to joblackburn and bheiner.
# This should always be committed as False; however, if you need to debug this unit test, set it temporarily to True
# @pytest.mark.ray_debug_mode(False)
@pytest.mark.parametrize(
    "experiment_config",
    [
        pytest.param("config/tasks/gymnasium/experiments/cartpole_v1_benchmark.yml", id="cartpole-v1-benchmark"),
    ],
)
def test_tasks(
    experiment_config,
    tmp_path,
    self_managed_ray,
):
    replace_hpc_dict = {
        "horizon": 10,
        "rollout_fragment_length": 10,
        "train_batch_size": 10,
        "sgd_minibatch_size": 10,
        "batch_mode": "complete_episodes",
        "num_workers": 1,
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_gpus_per_worker": 0,
        "num_gpus": 0,
        "num_sgd_iter": 30,
        "seed": 1,
    }

    args = parse_corl_args(["--cfg", experiment_config])
    experiment_class, experiment_file_validated = build_experiment(args)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, replace_hpc_dict, tmp_path)
    experiment_class.run_experiment(experiment_file_validated)
