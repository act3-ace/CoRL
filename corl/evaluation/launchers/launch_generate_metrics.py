"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Entry/Launch point to the generate_metrics process/segment of the evaluation framework.
"""
import itertools
import pathlib
import typing

import jsonargparse
import pandas as pd
import ray.cloudpickle as pickle

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.evaluation_artifacts import EvaluationArtifact_EvaluationOutcome, EvaluationArtifact_Metrics
from corl.evaluation.evaluation_outcome import EvaluationOutcome
from corl.evaluation.metrics.scenario_alert_generators import ScenarioAlertGenerators
from corl.evaluation.metrics.scenario_metric_generators import ScenarioMetricGenerators
from corl.evaluation.scene_processors import SceneProcessors
from corl.parsers.yaml_loader import load_file


def get_args(path: str | None = None) -> tuple[jsonargparse.Namespace, jsonargparse.Namespace]:
    """
    Obtain the running arguments for the generate_metrics segment of the evaluation framework

    Parameters
    ----------
    path : str
        optional path to a yaml config containing running arguments
    Returns:
        params: jsonargparse.Namespace
            namespace object that contains parsed running arguments
    """
    parser = jsonargparse.ArgumentParser(description="metrics processing")
    parser.add_argument("--cfg", help="the path to a json/yml file containing the running arguments", action=jsonargparse.ActionConfigFile)

    parser.add_class_arguments(EvaluationArtifact_EvaluationOutcome, "artifact_evaluation_outcome", instantiate=True)
    parser.add_class_arguments(EvaluationArtifact_Metrics, "artifact_metrics", instantiate=True)

    # Link location such that if metrics location isn't given, the location of evaluation_outcome will be used
    def func(source_object):
        return source_object.location.absolute_path

    parser.link_arguments("artifact_evaluation_outcome", "artifact_metrics.location", compute_fn=func, apply_on="instantiate")

    parser.add_argument("--alerts-config", type=str, help="config defining which alerts to process")
    parser.add_argument("--metrics-config", type=str, help="config defining which metrics to process")
    parser.add_argument(
        "--raise-on-error-alert", type=bool, default=True, help="If true will raise an exception when error alert encountered"
    )

    if path is None:
        args = parser.parse_args()
        instantiated = parser.instantiate_classes(args)
    else:
        args = parser.parse_path(path)
        instantiated = parser.instantiate_classes(args)

    return args, instantiated


def main(params: dict[str, typing.Any] | jsonargparse.Namespace):
    """
    Run the generate_metrics process

    Parameters
    ----------
    params: jsonargparse.Namespace
        namespace object that contains parsed running arguments
    """

    # ####################
    # Pull the EvaluationOutcome object

    outcome: EvaluationOutcome = params["artifact_evaluation_outcome"].location.load()

    # ####################
    # Build the metric generators

    metric_config = params["metrics_config"]
    if isinstance(metric_config, str):
        metric_config = load_file(params["metrics_config"])
    scenario_metric_generators = ScenarioMetricGenerators.from_dict(metric_config)

    # ####################
    # Build the alert generators

    alerts_config = params["alerts_config"]
    if isinstance(alerts_config, str):
        alerts_config = load_file(params["alerts_config"])
    elif alerts_config is None:
        alerts_config = {}
    scenario_alert_generators = ScenarioAlertGenerators.from_dict(alerts_config, params["raise_on_error_alert"])

    # ####################
    # Generate the Metrics

    test_cases = outcome.test_cases
    # calculate the dataframes we care about
    # self.loaded_legacy_dataframes = self.outcome.legacy_dataframes()
    if isinstance(test_cases, pd.DataFrame):
        test_cases = test_cases.reset_index()

    evaluation_data: list[EpisodeArtifact] = list(itertools.chain(*list(outcome.episode_artifacts.values())))

    processed_data = SceneProcessors.from_evaluation(
        evaluation_data=evaluation_data,
        scenario_metric_generators=scenario_metric_generators,
        scenario_alert_generators=scenario_alert_generators,
    )

    # ####################
    # Write the metrics

    if isinstance(params["artifact_metrics"].location, str):
        metrics_out = pathlib.Path(params["artifact_metrics"].location).joinpath(params["artifact_metrics"].file)

        with open(metrics_out, "wb") as wfp:
            pickle.dump(processed_data, wfp)
    elif isinstance(params["artifact_metrics"].location, pathlib.Path):
        metrics_out = params["artifact_metrics"].location.joinpath(params["artifact_metrics"].file)

        with open(metrics_out, "wb") as wfp:
            pickle.dump(processed_data, wfp)
    else:
        raise RuntimeError("Unknown type given in `out` field")


def pre_main():
    """
    calls gets current args and passes them to main
    """
    _, instantiated = get_args()
    main(instantiated)


if __name__ == "__main__":
    pre_main()
