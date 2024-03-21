"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Launch process for performing and visualizing an evaluation
"""
import typing

import jsonargparse

from corl.evaluation.evaluation_artifacts import EvaluationArtifact_Metrics, EvaluationArtifact_Visualization
from corl.evaluation.visualization.visualization import Visualization


def get_args(path=None):
    """
    Obtain the running arguments for the visualization segment of the evaluation framework
    """
    parser = jsonargparse.ArgumentParser(description="visualizer")
    parser.add_argument("--cfg", help="the path to a json/yml file containing the running arguments", action=jsonargparse.ActionConfigFile)

    parser.add_class_arguments(EvaluationArtifact_Metrics, "artifact_metrics", instantiate=True)
    parser.add_class_arguments(EvaluationArtifact_Visualization, "artifact_visualization", instantiate=True)

    # Link location such that if visualization isn't given, the metrics location will be used
    parser.link_arguments("artifact_metrics.location", "artifact_visualization.location", apply_on="instantiate")

    parser.add_argument("--visualizations", type=list[Visualization])

    args = parser.parse_args() if path is None else parser.parse_path(path)
    instantiate = parser.instantiate_classes(args)
    return args, instantiate


def main(instantiate: dict[str, typing.Any] | jsonargparse.Namespace):
    """
    Execute the visualization portion of the evaluation framework

    Parameters:
        args: jsonargparse.Namespace
            configurations for visualization
        instantiated: jsonargparse.Namespace
            instantiated arguments , contains  the actual classes to be run for visualization
    """

    for visual in instantiate["visualizations"]:
        visual.load(instantiate["artifact_metrics"], instantiate["artifact_visualization"])
        visual.visualize()


def pre_main():
    """
    calls gets current args and passes them to main
    """
    _, instantiated = get_args()
    main(instantiated)


if __name__ == "__main__":
    pre_main()
