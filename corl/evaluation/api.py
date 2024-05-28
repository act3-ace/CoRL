# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
"""

This module defines a python API for running CoRL's Evaluation Framework.

It also defines helper functions to support the reuse
of training configs, evaluation of multiple trained policies,
and creation of comparative plots of chosen Metrics. These functions
streamline visualization and analysis of comparative RL test assays.
"""

# pylint: disable=E0401

import re
import sys
import warnings
from pathlib import Path

import pandas as pd

from corl.evaluation.api_utils import add_required_metrics, construct_dataframe, construct_teams
from corl.evaluation.evaluation_artifacts import (
    EvaluationArtifact_EvaluationOutcome,
    EvaluationArtifact_Metrics,
    EvaluationArtifact_Visualization,
)
from corl.evaluation.launchers import launch_evaluate, launch_generate_metrics, launch_visualize
from corl.evaluation.recording.folder import FolderRecord
from corl.evaluation.runners.section_factories.test_cases.default_strategy import DefaultStrategy
from corl.evaluation.visualization.print import Print


def evaluate(
    task_config_path: str,
    checkpoint_path: str,
    output_path: str,
    experiment_config_path: str,
    launch_dir_of_experiment: str,
    platform_serializer_class: str,
    test_case_manager_config: dict | None = None,
    rl_algorithm_name: str | None = None,
    num_workers: int = 1,
):
    """
    This function is responsible for instantiating necessary arguments and then launching the first stage of CoRL's Evaluation Framework.

    Parameters
    ----------
    task_config_path: str
        The absolute path to the task_config used in training
    checkpoint_path: str
        The absolute path to the checkpoint from which the policy under evaluation will be loaded
    output_path: str
        The absolute path to the directory in which evaluation episode(s) data will be saved
    experiment_config_path: str
        The absolute path to the experiment config used in training
    launch_dir_of_experiment: str
        Paths in the experiment config are relative to the directory the corl.train_rl module is intended to be launched from.
        This string captures the path to the intended launch location of the experiment under evaluation.
    platform_serializer_class: PlatformSerializer
        The PlatformSerializer subclass capable of saving the state of the Platforms used in the evaluation environment
    test_case_manager_config: dict
        An optional map of TestCaseManager constructor arguments
    """

    # handle default test_case_manager
    if test_case_manager_config is None:
        test_case_manager_config = {"type": f"{DefaultStrategy.__module__}.{DefaultStrategy.__name__}", "config": {"num_test_cases": 3}}

    # construct teams map and test_cases for evaluation

    teams = construct_teams(experiment_config_path, launch_dir_of_experiment, checkpoint_path)

    # plugins
    plugins = {"platform_serialization": {"class_path": platform_serializer_class}}

    # recorders
    recorder = {"class_path": "corl.evaluation.recording.folder.Folder", "init_args": {"dir": output_path, "append_timestamp": False}}

    # instantiate eval objects
    task = {"config_yaml_file": task_config_path}

    # TODO: check functionality***
    # get rllib_engine config from task config

    trainer_cls = rl_algorithm_name if rl_algorithm_name is not None else "PPO"

    # handle multiprocessing
    rllib_engine = {"rllib": {"callbacks": [], "workers": num_workers, "trainer_cls": trainer_cls}}
    # task.experiment_parse.config['rllib_configs']['default'][1]["rollout_fragment_length"]="auto"

    # TODO: handle grid search***
    # # handle grid search seeds
    # if isinstance(task.experiment_parse.config['rllib_configs']['local'][0]['seed'], dict):
    #     task.experiment_parse.config['rllib_configs']['local'][0]['seed'] = 1

    # print("seed: " + str(task.experiment_parse.config['rllib_configs']['local'][0]['seed']) + '\n')

    cfg_path = None
    tmpdir_base = Path("/tmp")  # noqa: S108
    include_dashboard = False
    Path("/tmp/corl_evaluation/logging.yml")  # noqa: S108
    config = {
        "experiment": {
            "teams": teams,
            "task": task,
            "test_case_manager": test_case_manager_config,
            "plugins": plugins,
            "engine": rllib_engine,
            "recorders": [recorder],
        }
    }

    # call main
    launch_evaluate.main(cfg_path, tmpdir_base, config, include_dashboard=include_dashboard)  # type: ignore


def generate_metrics(evaluate_output_path: str, metrics_config: dict[str, dict | list]):
    """
    This function is responsible for instantiating necessary arguments and then launching the second stage of CoRL's Evaluation Framework.

    Parameters
    ----------
    evaluate_output_path: str
        The absolute path to the directory in which evaluation episodes' data was saved (from the initial 'evaluate' step of the
        Evaluation Framework)
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data
    """
    warnings.warn(
        "This function has been deprecated and will be removed in the future.\n\
            Users are encouraged to implement their own functions to interact with the `corl.evaluation.api.evaluate` function.",
        DeprecationWarning,
        stacklevel=2,
    )

    # define constructor args
    location = FolderRecord(absolute_path=evaluate_output_path)

    # TODO: enable evaluation without alerts
    alerts_config = {
        "world": [
            {
                "name": "Short Episodes",
                "metric": "rate_of_runs_lt_5steps",
                "scope": "evaluation",
                "thresholds": [{"type": "warning", "condition": {"operator": ">", "lhs": 0}}],
            }
        ]
    }

    raise_error_on_alert = True

    # instantiate eval objects
    outcome = EvaluationArtifact_EvaluationOutcome(location=location)
    metrics = EvaluationArtifact_Metrics(location=evaluate_output_path)

    # construct namespace dict
    namespace = {
        "artifact_evaluation_outcome": outcome,
        "artifact_metrics": metrics,
        "metrics_config": metrics_config,
        "alerts_config": alerts_config,
        "raise_on_error_alert": raise_error_on_alert,
    }

    launch_generate_metrics.main(namespace)


def visualize(evaluate_output_path: str):
    """
    This function is responsible for instantiating necessary arguments and then launching the third stage of CoRL's Evaluation Framework.

    Parameters
    ----------
    evaluate_output_path: str
        The absolute path to the directory in which evaluation episodes' data was saved (from the initial 'evaluate' step
        of the Evaluation Framework)
    """
    warnings.warn(
        "This function has been deprecated and will be removed in the future.\n\
            Users are encouraged to implement their own functions to interact with the `corl.evaluation.api.evaluate` function.",
        DeprecationWarning,
        stacklevel=2,
    )

    artifact_metrics = EvaluationArtifact_Metrics(location=evaluate_output_path)
    artifact_visualization = EvaluationArtifact_Visualization(location=evaluate_output_path)
    visualizations = [Print(event_table_print=True)]

    namespace = {"artifact_metrics": artifact_metrics, "artifact_visualization": artifact_visualization, "visualizations": visualizations}

    launch_visualize.main(namespace)


def run_evaluations(
    task_config_path: str,
    experiemnt_config_path: str,
    launch_dir_of_experiment: str,
    metrics_config: dict[str, dict | list],
    checkpoint_paths: list,
    output_paths: list,
    platfrom_serializer_class: str,
    test_case_manager_config: dict | None = None,
    visualize_metrics: bool = False,
    rl_algorithm_name: str | None = None,
    experiment_name: str = "",
    num_workers: int = 1,
):
    """
    This function is responsible for taking a list of checkpoint paths and iteratively running them through
    each stage of the Evaluation Framework (running Evaluation Episodes, processing results to generate Metrics,
    and optionally visualizing those Metrics).

    Parameters
    ----------
    task_config_path: str
        The absolute path to the directory in which evaluation episode(s) data will be saved
    experiemnt_config_path: str
        The absolute path to the experiment config used in training
    launch_dir_of_experiment: str
        Paths in the experiment config are relative to the directory the corl.train_rl module is intended to be launched from.
        This string captures the path to the intended launch location of the experiment under evaluation.
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data
    checkpoint_paths: list
        A list of path strings to each checkpoint, typically checkpoints are from a single training job
    output_paths: list
        A list of unique path strings at which each checkpoint's Evaluation Episodes data and Metrics will be stored.
        Must match the length of the checkpoint_paths list
    platfrom_serializer_class: PlatformSerializer
        The class object capable of storing data specific to the Platform type used in training
    test_case_manager_config: dict
        The kwargs passed to the TestCaseManager constructor. This must define the TestCaseStrategy class and its config
    visualize_metrics: bool
        A boolean to determine whether or not to run the Evaluation Framework's Visualize stage.
        If True, visualize is called.
    experiment_name: str
        The name of the experiment. Used for standard output progress updates.
    """
    warnings.warn(
        "This function has been deprecated and will be removed in the future.\n\
            Users are encouraged to implement their own functions to interact with the `corl.evaluation.api.evaluate` function.",
        DeprecationWarning,
        stacklevel=2,
    )

    kwargs = {}
    if test_case_manager_config:
        kwargs["test_case_manager_config"] = test_case_manager_config

    # rl algorithm
    if rl_algorithm_name:
        kwargs["rl_algorithm_name"] = rl_algorithm_name  # type: ignore

    kwargs["num_workers"] = num_workers  # type: ignore

    # run sequence of evaluation
    for index, ckpt_path in enumerate(checkpoint_paths):
        # print progress
        ckpt_num_regex = re.search(r"(checkpoint_)\w+", ckpt_path)
        ckpt_num = ckpt_path[ckpt_num_regex.start() : ckpt_num_regex.end()]  # type: ignore
        if experiment_name:
            print("\nExperiment: " + experiment_name)
        print("Evaluating " + ckpt_num + ". " + str(index + 1) + " of " + str(len(checkpoint_paths)) + " checkpoints.")

        # run evaluation episodes
        try:
            evaluate(
                task_config_path,
                ckpt_path,
                output_paths[index],
                experiemnt_config_path,
                launch_dir_of_experiment,
                platfrom_serializer_class,
                **kwargs,  # type: ignore
            )
        except SystemExit:
            print(sys.exc_info()[0])

        # generate evaluation metrics
        generate_metrics(output_paths[index], metrics_config)

        if visualize_metrics:
            # generate visualizations
            visualize(output_paths[index])


def run_one_evaluation(
    task_config_path: str,
    experiment_config_path: str,
    launch_dir_of_experiment: str,
    metrics_config: dict[str, dict | list],
    checkpoint_path: str,
    platfrom_serializer_class: str,
    test_case_manager_config: dict | None = None,
    rl_algorithm_name: str | None = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    """
    This function is responsible for taking a single checkpoint path and running it through
    each stage of the Evaluation Framework (running Evaluation Episodes, processing results
    to generate Metrics, and optionally visualizing those Metrics).

    Parameters
    ----------
    task_config_path: str
        The absolute path to the directory in which evaluation episode(s) data will be saved
    experiment_config_path: str
        The absolute path to the experiment config used in training
    launch_dir_of_experiment: str
        Paths in the experiment config are relative to the directory the corl.train_rl module is intended to be launched from.
        This string captures the path to the intended launch location of the experiment under evaluation.
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data
    checkpoint_path: str
        Path strings to the checkpoint
    platfrom_serializer_class: PlatformSerializer
        The class object capable of storing data specific to the Platform type used in training
    test_case_manager_config: dict
        The kwargs passed to the TestCaseManager constructor. This must define the TestCaseStrategy class and its config
    """
    warnings.warn(
        "This function has been deprecated and will be removed in the future.\n\
            Users are encouraged to implement their own functions to interact with the `corl.evaluation.api.evaluate` function.",
        DeprecationWarning,
        stacklevel=2,
    )

    kwargs = {}
    if test_case_manager_config:
        kwargs["test_case_manager_config"] = test_case_manager_config

    # rl algorithm
    if rl_algorithm_name:
        kwargs["rl_algorithm_name"] = rl_algorithm_name  # type: ignore

    kwargs["num_workers"] = num_workers  # type: ignore

    metrics_config = add_required_metrics(metrics_config)

    checkpoint_path += "/policies/{}/policy_state.pkl"
    exp_name = "single_episode__0__0"
    output_path = "/tmp/eval_results/" + exp_name  # noqa: S108

    # run evaluation episodes
    try:
        evaluate(
            task_config_path,
            checkpoint_path,
            output_path,
            experiment_config_path,
            launch_dir_of_experiment,
            platfrom_serializer_class,
            **kwargs,  # type: ignore
        )
    except SystemExit:
        print(sys.exc_info()[0])

    # generate evaluation metrics
    generate_metrics(output_path, metrics_config)

    experiment_to_eval_results_map = {}
    experiment_to_eval_results_map[exp_name] = {"output_paths": [output_path], "metadata": pd.DataFrame({"training_iteration": [0.0]})}
    return construct_dataframe(experiment_to_eval_results_map, metrics_config)
