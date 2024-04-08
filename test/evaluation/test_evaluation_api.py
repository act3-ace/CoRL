import glob
import os
from pathlib import Path

import ray

from corl.evaluation.api import evaluate
from corl.evaluation.runners.section_factories.test_cases.default_strategy import DefaultStrategy
from corl.evaluation.runners.section_factories.test_cases.tabular_strategy import TabularStrategy
from corl.evaluation.serialize_platforms import serialize_Docking_1d
from corl.test_utils import full_training
from corl.train_rl import build_experiment, parse_corl_args


def test_evaluate_tabular(tmp_path):
    # train 1d docking
    experiment_config = "config/tasks/docking_1d/experiments/docking_1d.yml"
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
    }  # TODO: stop criteria? ckpt freq?

    args = parse_corl_args(["--cfg", experiment_config])
    experiment_class, experiment_file_validated = build_experiment(args)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, ppo_rllib_config, tmp_path)
    experiment_class.config.tune_config["stop"]["training_iteration"] = 2
    experiment_class.config.tune_config["keep_checkpoints_num"] = 1
    experiment_class.run_experiment(experiment_file_validated)
    ray.shutdown()

    # Determine filename of the checkpoint
    ckpt_glob = list(tmp_path.glob("training/**/checkpoint_*"))
    ckpt_path = ckpt_glob[-1]

    # setup api variables
    launch_dir_of_experiment = str(tmp_path)
    task_config_path = str(Path(__file__).parent) + "/../../config/tasks/docking_1d/tasks/docking1d_task.yml"
    platform_serializer_class = f"{serialize_Docking_1d.__module__}.{serialize_Docking_1d.__name__}"

    tabular_test_case_manager_config = {
        "type": f"{TabularStrategy.__module__}.{TabularStrategy.__name__}",
        "config": {
            "config": {
                "data": "config/tasks/docking_1d/evaluation/test_cases_config/docking1d_tests.yml",
                "source_form": "FILE_YAML_CONFIGURATION",
                "randomize": False,
            }
        },
    }

    # eval output
    tabular_evaluation_ouput_dir = str(tmp_path) + "/eval_output_tabular"
    os.mkdir(tabular_evaluation_ouput_dir)
    assert os.path.exists(tabular_evaluation_ouput_dir)

    # TODO: add parameterized vars to test SAC + PPO
    # rl_alg_insert

    # launch evaluate
    evaluate(
        task_config_path,
        ckpt_path,
        tabular_evaluation_ouput_dir,
        experiment_config,
        launch_dir_of_experiment,
        platform_serializer_class,
        test_case_manager_config=tabular_test_case_manager_config,
        # rl_algorithm_name=rl_alg,
        num_workers=1,
    )

    # validate output (check if episode artifacts saved to file)
    episode_artifacts_path = str(tmp_path) + "/eval_output_tabular/test_case_*/*episode_artifact.pkl"
    episode_artifacts_paths = glob.glob(episode_artifacts_path)

    assert len(episode_artifacts_paths) == 3


def test_evaluate_default(tmp_path):
    # train 1d docking
    experiment_config = "config/tasks/docking_1d/experiments/docking_1d.yml"
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
    }  # TODO: stop criteria? ckpt freq?

    args = parse_corl_args(["--cfg", experiment_config])
    experiment_class, experiment_file_validated = build_experiment(args)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, ppo_rllib_config, tmp_path)
    experiment_class.config.tune_config["stop"]["training_iteration"] = 2
    experiment_class.config.tune_config["keep_checkpoints_num"] = 1
    experiment_class.run_experiment(experiment_file_validated)
    ray.shutdown()

    # Determine filename of the checkpoint
    ckpt_glob = list(tmp_path.glob("training/**/checkpoint_*"))
    ckpt_path = ckpt_glob[-1]

    # setup api variables
    launch_dir_of_experiment = str(tmp_path)
    task_config_path = str(Path(__file__).parent) + "/../../config/tasks/docking_1d/tasks/docking1d_task.yml"
    platform_serializer_class = f"{serialize_Docking_1d.__module__}.{serialize_Docking_1d.__name__}"

    default_test_case_manager_config = {"type": f"{DefaultStrategy.__module__}.{DefaultStrategy.__name__}", "config": {"num_test_cases": 4}}

    # eval output
    default_evaluation_ouput_dir = str(tmp_path) + "/eval_output_default"
    os.mkdir(default_evaluation_ouput_dir)
    assert os.path.exists(default_evaluation_ouput_dir)

    # TODO: add parameterized vars to test SAC + PPO
    # rl_alg_insert

    # launch evaluate
    evaluate(
        task_config_path,
        ckpt_path,
        default_evaluation_ouput_dir,
        experiment_config,
        launch_dir_of_experiment,
        platform_serializer_class,
        test_case_manager_config=default_test_case_manager_config,
        # rl_algorithm_name=rl_alg,
        num_workers=1,
    )

    # validate output (check if episode artifacts saved to file)
    episode_artifacts_path = str(tmp_path) + "/eval_output_default/test_case_*/*episode_artifact.pkl"
    episode_artifacts_paths = glob.glob(episode_artifacts_path)

    assert len(episode_artifacts_paths) == 4
