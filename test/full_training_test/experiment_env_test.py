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

from corl.experiments.base_experiment import ExperimentParse
from corl.experiments.benchmark_experiment import BenchmarkExperiment
from corl.test_utils import full_training
from corl.train_rl import ExperimentFileParse, merge_cfg_and_args, parse_corl_args, parse_experiments_yml


def build_experiment_benchmark(args):
    """
    handles the argparsing for CoRL, this function allows alternate arguments to be
    input to allow unit tests to use the same parsing code

    Args:
        alternate_argv (typing.Optional[typing.Sequence[str]], optional): _description_. Defaults to None.

    Returns:
        _type_: fully parsed arguments
    """
    cfg = parse_experiments_yml(config_filename=args.cfg)
    cfg = merge_cfg_and_args(cfg, args)
    experiment_file_validated = ExperimentFileParse(**cfg)
    experiment_parse = ExperimentParse(**experiment_file_validated.config)
    experiment_parse.config["benchmark_episodes"] = 1
    experiment_parse.experiment_class.process_cli_args(experiment_parse.config, experiment_file_validated)
    experiment_class = BenchmarkExperiment(**experiment_parse.config)

    return experiment_class, experiment_file_validated


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
def test_experiments(experiment_config, test_rllib_config, tmp_path):
    args = parse_corl_args(["--cfg", experiment_config])
    experiment_class, experiment_file_validated = build_experiment_benchmark(args)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, test_rllib_config, tmp_path)
    experiment_class.config.profile = False
    experiment_class.run_experiment(experiment_file_validated)


def test_configs_validated(request, all_experiments):
    tested_configs = [Path(test_tasks_args.values[0]) for test_tasks_args in request.module.test_experiments.pytestmark[0].args[1]]
    exempt_experiments = [
        # tested in offline_training_test
        Path("config/tasks/pong/experiments/pong_marwil.yml")
    ]
    missing_tests = set(all_experiments) - set(tested_configs) - set(exempt_experiments)
    assert len(missing_tests) == 0
