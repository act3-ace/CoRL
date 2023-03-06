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


# Adjustments to the configuration files used in this test to match the current baseline is authorized, provided you make a post on
# MatterMost aaco-ai-agents with notifications to joblackburn and bheiner that the change was made.
# All other changes, such as commenting out tests or disabling inference or evaluation requires coordination on the aaco-ai-agents
# channel of MatterMost with notifications to joblackburn and bheiner.
# This should always be committed as False; however, if you need to debug this unit test, set it temporarily to True
# @pytest.mark.ray_debug_mode(False)
@pytest.mark.parametrize(
    'experiment_config',
    [
        pytest.param(
            'config/tasks/openai_gym/experiments/cartpole_v1_benchmark.yml',
            id='cartpole-v1-benchmark'
        ),
    ],
)
def test_tasks(
    experiment_config,
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

    replace_hpc_dict = {
        'horizon': 10,
        'rollout_fragment_length': 10,
        'train_batch_size': 10,
        'sgd_minibatch_size': 10,
        'batch_mode': 'complete_episodes',
        'num_workers': 1,
        'num_cpus_per_worker': 1,
        'num_envs_per_worker': 1,
        'num_cpus_for_driver': 1,
        'num_gpus_per_worker': 0,
        'num_gpus': 0,
        'num_sgd_iter': 30,
        'seed': 1
    }

    experiment_class.config.rllib_configs["local"] = replace_hpc_dict
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
    experiment_class.run_experiment(args)
