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

This module defines defines helper functions to support the reuse
of training configs, evaluation of multiple trained policies,
and creation of comparative plots of chosen Metrics. These functions
streamline visualization and analysis of comparative RL test assays.
"""

# pylint: disable=E0401

import os
import pickle
from glob import glob

import jsonlines
import pandas as pd
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis

from corl.evaluation.loader.policy_checkpoint import PolicyCheckpoint
from corl.evaluation.metrics.aggregators.criteria_rate import CriteriaRate
from corl.evaluation.metrics.generators.meta.episode_length import EpisodeLength_Steps
from corl.parsers.yaml_loader import load_file


def construct_teams(experiment_config_path: str, launch_dir_of_experiment: str, checkpoint_path: str) -> dict[str, list]:
    """
    This function is responsible for creating the Teams object required by the Evaluation Framework. It uses the experiment
    config file from training to get agent and platform info required by the Teams class. Use of this function assumes the user wishes
    to replicate the training environment for evaluation episodes.

    Parameters
    ----------
    experiment_config_path: str
        The absolute path to the experiment config used in training
    launch_dir_of_experiment: str
        Paths in the experiment config are relative to the directory the corl.train_rl module is intended to be launched from.
        This string captures the path to the intended launch location of the experiment under evaluation.
    checkpoint_path: str
        The absolute path to the checkpoint from which each agent's policy will be loaded.
        The directory titled the agent's name will be filled in programmatically during this function.
        An example of the format of this string is: '/path/to/experiment/output/checkpoint_00005/policies/{}/policy_state.pkl'.

    Returns
    -------
    Teams: corl.evaluation.runners.section_factories.teams.Teams
        Maintains a map of every platform, policy_id, agent_config, and name for each entity present in the training environment
    """

    # parse experiment config
    experiment_config = load_file(experiment_config_path)

    assert "agent_config" in experiment_config
    assert "platform_config" in experiment_config
    agents_config = experiment_config["agent_config"]
    platforms_config = experiment_config["platform_config"]

    # populate teams based on experiment config
    teams_platforms_config = []
    teams_agents_config = []
    for index, agent_info in enumerate(agents_config):
        agent_name = agent_info["name"]
        platforms = agent_info["platforms"]
        agent_config_path = agent_info["config"]
        policy_config_path = agent_info["policy"]
        platform_name = platforms_config[index]["name"]
        platform_config_path = platforms_config[index]["config"]

        # handle relative paths from experiment config
        agent_config_path = os.path.join(launch_dir_of_experiment, agent_config_path)
        platform_config_path = os.path.join(launch_dir_of_experiment, platform_config_path)
        policy_config_path = os.path.join(launch_dir_of_experiment, policy_config_path)

        # from pathlib import Path
        agent_loader = {
            "class_path": f"{PolicyCheckpoint.__module__}.{PolicyCheckpoint.__name__}",
            "init_args": {"trained_agent_id": agent_name, "checkpoint_filename": checkpoint_path},
        }
        teams_platforms_config.append({"name": platform_name, "config": platform_config_path})
        teams_agents_config.append(
            {
                "name": agent_name,
                "config": agent_config_path,
                "platforms": platforms,
                "policy": policy_config_path,
                "agent_loader": agent_loader,
            }
        )

    return {"platform_config": teams_platforms_config, "agent_config": teams_agents_config}


def add_required_metrics(metrics_config: dict[str, dict | list]) -> dict[str, dict | list]:
    """
    This helper function is responsible for adding in a few Metrics that are required by the Evaluation Framework. This is to
    simplify the configuration process for the user.

    Parameters
    ----------
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data

    Returns
    -------
    metrics_config: dict
        The mutated metrics_config
    """

    # add required metrics to metrics_config
    episode_length_metric = {
        "name": "EpisodeLength(Steps)",
        "functor": f"{EpisodeLength_Steps.__module__}.{EpisodeLength_Steps.__name__}",
        "config": {"description": "episode length of test case rollout in number of steps"},
    }
    episode_length_alert_metric = {
        "name": "rate_of_runs_lt_5steps",
        "functor": f"{CriteriaRate.__module__}.{CriteriaRate.__name__}",
        "config": {
            "description": "alert metric to see if any episode length is less than 5 steps",
            "metrics_to_use": "EpisodeLength(Steps)",
            "scope": {"type": "corl.evaluation.metrics.scopes.from_string", "config": {"name": "evaluation"}},
            "condition": {"operator": "<", "lhs": 5},
        },
    }

    if "world" in metrics_config:
        assert isinstance(metrics_config["world"], list), "'world' metrics in metrics_config must be list of dicts"
        metrics_config["world"].append(episode_length_metric)
        metrics_config["world"].append(episode_length_alert_metric)
    else:
        metrics_config["world"] = [episode_length_metric, episode_length_alert_metric]

    return metrics_config


def parse_metrics_config(metrics_config: dict[str, dict | list]) -> dict[str, dict | list]:
    """
    This function is responsible for walking the metrics config and creating a dictionary of Metric
    names present in the metrics config.

    Parameters
    ----------
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data

    Returns
    -------
    metrics_names: dict
        A dictionary resembling the metrics_config, but containing only Metric names
    """
    # collect metric names
    metrics_names = {}  # type: ignore
    if "world" in metrics_config:
        metrics_names["world"] = []
        for world_metric in metrics_config["world"]:
            metrics_names["world"].append(world_metric["name"])

    if "agent" in metrics_config:
        # TODO: support agent-specific metrics
        metrics_names["agent"] = {}
        if "__default__" in metrics_config["agent"]:
            metrics_names["agent"]["__default__"] = []
            for agent_metric in metrics_config["agent"]["__default__"]:  # type: ignore
                metrics_names["agent"]["__default__"].append(agent_metric["name"])

    return metrics_names


def construct_dataframe(results: dict[str, dict], metrics_config: dict[str, dict | list]) -> pd.DataFrame:  # pylint: disable=R0914
    """
    This function is responsible for parsing Metric data from the results of Evaluation Episodes.
    It collects values from all Metrics included in the metrics_config into a single pandas.DataFrame,
    which is returned by the function.

    Parameters
    ----------
    results: dict[str, dict]
        A map of experiment names to Evaluation result locations and training duration metadata
    metrics_config: dict
        The nested structure defining the Metrics to be instantiated and calculated from Evaluation Episode data

    Returns
    -------
    dataframe: pandas.DataFrame
        A DataFrame containing Metric values collected for each checkpoint of each experiment by row
    """
    metric_names = parse_metrics_config(metrics_config)
    columns = ["experiment", "experiment_index", "evaluation_episode_index", "training_iteration", "agent_name", "agent"]

    # need to parse each metrics.pkl file + construct DataFrame
    dataframes = []
    for trial_id in results:  # pylint: disable=R1702
        data = []
        output_paths = results[trial_id]["output_paths"]
        training_metadata_df = results[trial_id]["metadata"]

        # collect metric values per experiment
        for output_path in output_paths:
            # collect agent metrics per ckpt
            with open(output_path + "/metrics.pkl", "rb") as metrics_file:
                metrics = pickle.load(metrics_file)  # noqa: S301

            for agent_name, participant in metrics.participants.items():
                episode_events = list(participant.events)

                experiment_name, experiment_index = trial_id.split("__")
                checkpoint_num = int(output_path.split("_")[-1])  # TODO: want better way to get checkpoint num data here

                experiment_agent = experiment_name + "_" + agent_name

                for evaluation_episode_index, event in enumerate(episode_events):
                    # aggregate trial data (single dataframe entry)
                    row = [experiment_name, experiment_index, evaluation_episode_index, checkpoint_num, agent_name, experiment_agent]

                    # collect agent's metric values on each trial in eval
                    for metric_name in metric_names["agent"]["__default__"]:  # type: ignore
                        assert metric_name in event.metrics, f"{metric_name} not an available metric!"
                        # add metric value to row
                        if hasattr(event.metrics[metric_name], "value"):
                            row.append(event.metrics[metric_name].value)
                        elif hasattr(event.metrics[metric_name], "arr"):
                            row.append(event.metrics[metric_name].arr)
                        else:
                            raise ValueError("Metric must have attribute 'value' or 'arr'")
                        # add metric name to columns
                        if metric_name not in columns:
                            columns.append(metric_name)

                    # add row to dataset [experiment name, iteration, num_episodes, num_interactions, episode ID/trial, **custom_metrics]
                    data.append(row)

                # collect world metrics per ckpt
                ...

        # create experiment dataframe
        expr_dataframe = pd.DataFrame(data, columns=columns)
        # join metadata on training iteration / checkpoint_num
        expr_dataframe = pd.merge(expr_dataframe, training_metadata_df, how="inner", on="training_iteration")
        # add to experiment dataframes collection
        dataframes.append(expr_dataframe)

    # collect experiment dataframes into single DataFrame
    return pd.concat(dataframes).reset_index()


def checkpoints_list_from_experiment_analysis(
    experiment_analysis: ExperimentAnalysis, output_dir: str, experiment_name: str, trial_index: int = 0
) -> tuple[list[str], list[str]]:
    """
    This function is responsible for compiling a list of paths to each checkpoint in an experiment.
    This acts as a helper function for when users want to evaluate a series of checkpoints from a single training job.

    Parameters
    ----------
    experiment_analysis: ExperimentAnalysis
        The ExperimentAnalysis object containing checkpoints and results of a single training job
    output_dir: str
        The absolute path to the directory that will hold the results from Evaluation Episodes
    experiment_name: str
        The name of the experiment under evaluation

    Returns
    -------
    checkpoint_paths: list
        A list of absolute paths to each checkpoint file found in the provided training_output_path
    output_dir_paths: list
        A list of absolute paths used to determine where to save Evaluation Episode data for each checkpoint
    """

    # create ExperimentAnalysis object to handle Trial Checkpoints
    trial = experiment_analysis.trials[trial_index]
    ckpt_paths = experiment_analysis.get_trial_checkpoints_paths(trial, "training_iteration")

    output_dir_paths = []
    checkpoint_paths = []
    for path, trainig_iteration in ckpt_paths:
        # TODO: verify policy_state desired over algorithm_state!
        if os.path.isdir(path):
            path += "/policies/{}/policy_state.pkl"  # noqa: PLW2901

        checkpoint_paths.append(path)
        output_path = output_dir + "/" + experiment_name + "/" + "checkpoint_" + str(trainig_iteration)
        output_dir_paths.append(output_path)

    return checkpoint_paths, output_dir_paths


def extract_metadata_from_experiment_analysis(experiment_analysis: ExperimentAnalysis, trial_index: int = 0) -> pd.DataFrame:
    """
    This function is responsible for collecting training duration information from the ExperimentAnalysis object.
    This function currently collects the number of training iterations, Episodes, environment interactions, and
    walltime seconds that had occurred at the time the checkpoint was created.

    Parameters
    ----------
    experiment_analysis : ExperimentAnalysis
        Ray Tune experiment analysis object
    trial_index : int
        the index of the trial of interest

    Returns
    -------
    training_meta_data: pandas.DataFrame
        A collection of training duration information for each provided checkpoint
    """

    # assumes one trial per training job
    trial = experiment_analysis.trials[trial_index]
    df = experiment_analysis.trial_dataframes[trial.logdir]
    training_meta_data = df[["training_iteration", "timesteps_total", "episodes_total", "time_total_s"]]
    # add trial index
    training_meta_data["trial_index"] = [trial_index] * training_meta_data.shape[0]

    return training_meta_data


def extract_metadata_from_result_file(trial_output_path: str) -> pd.DataFrame:
    """
    This function is responsible for collecting training duration information from the ExperimentAnalysis object.
    This function currently collects the number of training iterations, Episodes, environment interactions, and
    walltime seconds that had occurred at the time the checkpoint was created.

    Parameters
    ----------
    trial_output_path : str
        path to trial output

    Returns
    -------
    training_meta_data: pandas.DataFrame
        A collection of training duration information for each provided checkpoint
    """
    # get training iters of trial checkpoints
    training_iterations = []
    ckpt_paths = glob(trial_output_path + "/checkpoint*")
    for path in ckpt_paths:
        training_iterations.append(get_training_iter_from_ckpt_path(path))  # noqa: PERF401

    # find result.json
    result_file = glob(trial_output_path + "/result.json")[0]

    # parse result.json for metadata of checkpoints
    metadata: dict[str, list] = {
        "training_iteration": [],
        "timesteps_total": [],
        "episodes_total": [],
        "time_total_s": [],
    }

    with jsonlines.open(result_file) as jsonlines_reader:
        for training_iteration_data in jsonlines_reader:
            # search result.json file for training iterations in which checkpoints were saved
            training_iteration = training_iteration_data["training_iteration"]
            if training_iteration in training_iterations:
                # data is relevant to a saved checkpoint
                # collect and store the desired metadata
                metadata["training_iteration"].append(training_iteration)
                metadata["timesteps_total"].append(training_iteration_data["timesteps_total"])
                metadata["episodes_total"].append(training_iteration_data["episodes_total"])
                metadata["time_total_s"].append(training_iteration_data["time_total_s"])

    return pd.DataFrame(metadata)


def get_checkpoints_paths(trial_output_path: str, trial_name: str, evaluation_ouput_path: str) -> tuple[list[str], list[str]]:
    """
    Function responsible for finding checkpoint paths in a trial output dir and returning them.
    It also creates a list of each trial's evaluation output dir and returns them.
    """

    ckpt_paths = glob(trial_output_path + "/checkpoint*")

    output_dir_paths = []
    checkpoint_paths = []
    for path in ckpt_paths:
        training_iteration = get_training_iter_from_ckpt_path(path)
        if training_iteration > 0:
            # skip 0th checkpoint if included - no metadata available in result.json file
            checkpoint_paths.append(path)
            output_path = evaluation_ouput_path + "/" + trial_name + "/" + "checkpoint_" + str(training_iteration)
            output_dir_paths.append(output_path)

    return checkpoint_paths, output_dir_paths


def get_training_iter_from_ckpt_path(ckpt_path: str) -> int:
    return int(ckpt_path.split("/")[-1].split("_")[-1])
