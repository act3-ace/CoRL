"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import pathlib

import pytest
import ray
import yaml

from corl.train_rl import parse_corl_args
from corl.parsers.yaml_loader import load_file
from corl.experiments.base_experiment import ExperimentParse


ppo_rllib_config = {
    'horizon': 10,
    'rollout_fragment_length': 10,
    'train_batch_size': 10,
    'sgd_minibatch_size': 10,
    'num_workers': 1,
    'num_cpus_per_worker': 1,
    'num_envs_per_worker': 1,
    'num_cpus_for_driver': 1,
    'num_gpus_per_worker': 0,
    'num_gpus': 0,
    'num_sgd_iter': 30,
    'seed': 1
}

sac_rllib_config = {
    'horizon': 10,
    'rollout_fragment_length': 10,
    'train_batch_size': 10,
    'num_workers': 1,
    'num_cpus_per_worker': 1,
    'num_envs_per_worker': 1,
    'num_cpus_for_driver': 1,
    'num_gpus_per_worker': 0,
    'num_gpus': 0,
    'seed': 1
}

# Adjustments to the configuration files used in this test to match the current baseline is authorized, provided you make a post on
# MatterMost aaco-ai-agents with notifications to joblackburn and bheiner that the change was made.
# All other changes, such as commenting out tests or disabling inference or evaluation requires coordination on the aaco-ai-agents
# channel of MatterMost with notifications to joblackburn and bheiner.
# This should always be committed as False; however, if you need to debug this unit test, set it temporarily to True
# @pytest.mark.ray_debug_mode(False)
@pytest.mark.parametrize(
    'experiment_config, eval_config, test_rllib_config',
    [
        pytest.param(
            'config/tasks/pong/experiments/pong_commander.yml',
            'config/tasks/pong/evaluation/launch/commander_pong_evaluate.yml',
            ppo_rllib_config,
            id='pong commander ppo eval'
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
    # optional_debuggable_ray = False
    # if optional_debuggable_ray:
    #     ray_config['local_mode'] = True
    # else:
    #     ray_config['ignore_reinit_error'] = True

    args = parse_corl_args(["--cfg", experiment_config])
    config = load_file(config_filename=args.config)

    # print(config)
    experiment_parse = ExperimentParse(**config)
    experiment_class = experiment_parse.experiment_class(**experiment_parse.config)

    experiment_class.config.rllib_configs["local"].update(test_rllib_config)
    if "model" in experiment_class.config.rllib_configs["local"]:
        experiment_class.config.rllib_configs["local"]["model"].reset()

    experiment_class.config.ray_config['ignore_reinit_error'] = True
    if "_temp_dir" in experiment_class.config.ray_config:
        del experiment_class.config.ray_config["_temp_dir"]

    experiment_class.config.env_config["output_path"] = str(tmp_path / "training")

    experiment_class.config.tune_config['stop']['training_iteration'] = 1
    experiment_class.config.tune_config['local_dir'] = str(tmp_path / "training")
    experiment_class.config.tune_config['checkpoint_freq'] = 1
    experiment_class.config.tune_config['max_failures'] = 1
    args.compute_platform = "local"
    experiment_class.run_experiment(args)


    # Determine filename of the checkpoint
    checkpoint_glob = list(tmp_path.glob('training/**/checkpoint_000001/'))
    assert len(checkpoint_glob) == 1
    checkpoint = checkpoint_glob[0]
    print(checkpoint_glob)
    policy_file = checkpoint / "policies" / "paddle_ctrl" / "policy_state.pkl"
    print(policy_file)
    assert policy_file.exists()

    eval_launch_config = load_file(eval_config)
    eval_launch_config["teams"]["agent_config"][0]["agent_loader"]["init_args"]["checkpoint_filename"] = str(policy_file)

    import pprint

    pprint.pprint(eval_launch_config)

    tmp_launch_file = tmp_path / "eval_launch.yml"

    with open(tmp_launch_file, 'w') as yaml_file:
        yaml.dump(eval_launch_config, yaml_file, default_flow_style=False)

    assert tmp_launch_file.exists()

    from corl.evaluation.launchers.launch_evaluate import pre_main

    pre_main(alternate_argv=["--cfg", str(tmp_launch_file)])


