"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import pprint

import pytest
import yaml

from corl.parsers.yaml_loader import load_file
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
    "num_sgd_iter": 30,
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
    "experiment_config, eval_config, test_rllib_config",
    [
        pytest.param(
            "config/tasks/pong/experiments/pong.yml",
            "config/tasks/pong/evaluation/launch/base_pong_evaluate.yml",
            ppo_rllib_config,
            id="pong commander ppo eval",
        ),
        pytest.param(
            "config/tasks/pong/experiments/pong.yml",
            "config/tasks/pong/evaluation/launch/base_pong_inference.yml",
            ppo_rllib_config,
            id="pong commander ppo inference",
        ),
    ],
)
def test_tasks(
    experiment_config,
    eval_config,
    test_rllib_config,
    tmp_path,
    self_managed_ray,
):
    # optional_debuggable_ray = True
    # if optional_debuggable_ray:
    #     ray_config['local_mode'] = True
    # else:
    #     ray_config['ignore_reinit_error'] = True

    args = parse_corl_args(["--cfg", experiment_config])
    experiment_class, experiment_file_validated = build_experiment(args)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, test_rllib_config, tmp_path)
    experiment_class.run_experiment(experiment_file_validated)

    # Determine filename of the checkpoint
    checkpoint_glob = list(tmp_path.glob("training/**/checkpoint_000000/"))
    assert len(checkpoint_glob) == 1
    checkpoint = checkpoint_glob[0]
    print(checkpoint_glob)
    policy_file0 = checkpoint / "policies" / "paddle0_ctrl"
    policy_file1 = checkpoint / "policies" / "paddle1_ctrl"

    eval_launch_config = load_file(eval_config)

    teams = eval_launch_config.get("teams", eval_launch_config.get("experiment", {}).get("teams"))
    assert teams is not None

    teams["agent_config"][0]["agent_loader"]["init_args"]["checkpoint_filename"] = str(policy_file0.parent.parent)
    teams["agent_config"][1]["agent_loader"]["init_args"]["checkpoint_filename"] = str(policy_file1.parent.parent)

    pprint.pprint(eval_launch_config)  # noqa: T203

    tmp_launch_file = tmp_path / "eval_launch.yml"

    with open(tmp_launch_file, "w") as yaml_file:
        yaml.dump(eval_launch_config, yaml_file, default_flow_style=False)

    assert tmp_launch_file.exists()

    from corl.evaluation.launchers.launch_evaluate import pre_main

    pre_main(alternate_argv=["--cfg", str(tmp_launch_file)])
