"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Entry / Launch point to run the evaluation framework processes, evaluation , generate_metrics , visualize back to back .
There are also various storage options
"""
from pathlib import Path
from typing import Any

import jsonargparse

from corl.evaluation.evaluation_artifacts import (
    EvaluationArtifact_EvaluationOutcome,
    EvaluationArtifact_Metrics,
    EvaluationArtifact_Visualization,
)
from corl.evaluation.launchers.base_eval import BaseProcessor, EvalConfig
from corl.evaluation.launchers.launch_evaluate import get_args as eval_get_args
from corl.evaluation.launchers.launch_evaluate import load_config as eval_load_config
from corl.evaluation.launchers.launch_evaluate import main as evaluate_main
from corl.evaluation.launchers.launch_generate_metrics import get_args as gen_metrics_get_args
from corl.evaluation.launchers.launch_generate_metrics import main as generate_metrics_main
from corl.evaluation.launchers.launch_storage import get_args as storage_get_args
from corl.evaluation.launchers.launch_storage import main as storage_main
from corl.evaluation.launchers.launch_visualize import get_args as visualize_get_args
from corl.evaluation.launchers.launch_visualize import main as visualize_main
from corl.evaluation.loader.policy_checkpoint import PolicyCheckpoint
from corl.evaluation.recording.folder import FolderRecord
from corl.evaluation.recording.i_recorder import IRecord
from corl.evaluation.runners.iterate_test_cases import ConfigSaver


class PipelineProcessor(BaseProcessor):
    def __init__(self, run_params: jsonargparse.Namespace, agent_checkpoints: dict[str, Path], **kwargs):
        super().__init__(**kwargs)
        self.run_params = run_params
        self.agent_checkpoints = agent_checkpoints

    def __call__(self, config: EvalConfig, results: Any):
        print("\n")
        print("Evaluation portion complete")
        use_eval_record = results[0]

        gen_metrics_params, metrics_artifact = gen_metrics_args(self.run_params, use_eval_record)

        print("\n")
        print("executing generate_metrics segment")

        generate_metrics_main(gen_metrics_params)

        print("\n")
        print("generate_metrics segment complete")

        _, visualize_params_instantiated, artifact_visualize = visualize_args(self.run_params, metrics_artifact)

        print("\n")
        print("executing visualization segment")

        visualize_main(visualize_params_instantiated)

        print("completed visualization segment")
        print("Evaluation framework pipeline complete")

        # collect visualization locations from visualize_params_instantiated

        if self.run_params.storage_config:
            artifact_location_config = {
                "eval_data_location": Path(str(use_eval_record.absolute_path)),
                "metrics_file_location": Path(str(metrics_artifact.location)).joinpath(str(metrics_artifact.file)),
                "visualizations_location": Path(str(artifact_visualize.location)),
                "agent_checkpoints": self.agent_checkpoints,
            }

            storage_instantiated_args = storage_args(self.run_params, artifact_location_config)
            storage_main(storage_instantiated_args)


def get_run_args() -> jsonargparse.Namespace:
    """
    Obtain the appropriate config arguments to run the evaluation framework pipeline.

    Returns
    -------
    run_params: jsonargparse.Namespace
        running configurations for the pipeline.
    """

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--cfg", action=jsonargparse.ActionConfigFile, help="path to yaml file containing run arguments")
    parser.add_argument("--eval_config", type=str, help="path to evaluation launch config")
    parser.add_argument("--gen_metrics_config", type=str, help="path to generate_metrics launch config")
    parser.add_argument("--visualize_config", type=str, help="path to visualize launch config")
    parser.add_argument("--storage_config", type=str, default=None, help="path to acedt bottle config")

    return parser.parse_args()


def get_evaluation_save(instantiated: type[jsonargparse.Namespace]) -> FolderRecord:
    """
    Obtain the location at which the evaluation segment recorded its results to. Picks up the first save location.

    Parameters:
        args: jsonargparse.Namespace
            contains the uninstantiated arguments from the evaluation segment.

    Returns:
        FolderRecord
            The first save location used in the evaluation segment to record evaluation results.
    """
    return instantiated["recorders"][0]  # type: ignore[index]


def get_agent_checkpoints(instantiated_args: jsonargparse.Namespace) -> dict[str, Path]:
    """
    From the evaluation segment retrieve the agent checkpoint locations.

    instantiated_args: jsonargparse.Namespace
        the instantiated classes arguments from the evaluation segment that setups the actual evaluation process

    Returns:
        agent_checkpoints: Dict[str,str]
            dictionary holding the agent checkpoints stored as agent_name to location.

    """
    agent_checkpoints = {}
    teams_map = instantiated_args["teams"]
    for agent in teams_map.agent_config:
        if isinstance(agent.agent_loader, PolicyCheckpoint):
            agent_checkpoints[agent.name] = Path(agent.agent_loader.checkpoint_filename)
    return agent_checkpoints


def evaluation_args(run_params: jsonargparse.Namespace) -> tuple[jsonargparse.Namespace, dict[str, Path]]:
    """
    Retrieve all run arguments to execute evaluation process and return the first save location for eval output.

    Parameters
    ----------
    run_params: jsonargparse.Namespace
        The running parameters for the pipeline

    Returns
    -------
    instantiated_args: jsonargparse.Namespace
        The instantiated classes needed to run the evaluation process
    eval_save_loc: str
        the first location to where the evaluation

    """

    cfg_file = run_params.eval_config

    instantiated_args, _ = eval_get_args(path=cfg_file)

    agent_checkpoints = get_agent_checkpoints(instantiated_args)

    return instantiated_args, agent_checkpoints


def gen_metrics_args(run_params: jsonargparse.Namespace, eval_record: IRecord) -> tuple[jsonargparse.Namespace, EvaluationArtifact_Metrics]:
    """
    Retrieve the running arguments for the generate_metrics process. Reconfigures locations as appropriate.

    Parameters
    ----------
    run_params: jsonargparse.Namespace
        configurations for the pipeline
    evaluation_outcome: EvaluationArtifact_EvaluationOutcome
        location of evaluation data output

    Returns
    -------
    args: jsonargparse.Namespace
        configurations for the gen_metrics process
    metrics_file_location:
        location of the metrics pkl file
    """

    gen_metrics_cfg = run_params.gen_metrics_config

    _, args = gen_metrics_get_args(gen_metrics_cfg)

    # by default grab the first record

    metrics_artifact = EvaluationArtifact_Metrics(str(eval_record.absolute_path))
    eval_artifact = EvaluationArtifact_EvaluationOutcome(location=eval_record)

    args["artifact_metrics"] = metrics_artifact
    args["artifact_evaluation_outcome"] = eval_artifact

    return args, metrics_artifact


def visualize_args(
    run_params: jsonargparse.Namespace, metrics_artifact: EvaluationArtifact_Metrics
) -> tuple[jsonargparse.Namespace, jsonargparse.Namespace, EvaluationArtifact_Visualization]:
    """
    Retrieve the running arguments for the visualize process. Reconfigures locations as appropriate.

    Parameters
    ----------
    run_params: jsonargparse.Namespace
        configurations for the pipeline
    metrics_artifact: EvaluationArtifact_Metrics
        Meta data of the generate metrics process
    Returns
    -------
    args: jsonargparse.Namespace
        configurations for the visualize process
    """

    visualize_cfg = run_params.visualize_config

    args, instantiate = visualize_get_args(visualize_cfg)

    common_folder = metrics_artifact.location
    visualization_folder = Path(common_folder).joinpath("visualizations")
    if not visualization_folder.is_dir():
        visualization_folder.mkdir()
    artifact_visualize = EvaluationArtifact_Visualization(str(visualization_folder))

    instantiate["artifact_metrics"] = metrics_artifact
    instantiate["artifact_visualization"] = artifact_visualize

    return args, instantiate, artifact_visualize


def storage_args(run_params, artifacts_config):
    """
    Retrieve the running arguments for the storage process. Reconfigures locations as appropriate.

    Parameters
    ----------
    run_params: jsonargparse.Namespace
        configurations for the pipeline
    artifacts_config: dict
        contains locations of all the artifacts from the evaluation framework pipeline.

    Returns
    -------
    instantiated: jsonargparse.Namespace
        configurations for the storage process
    """

    storage_instantiated = storage_get_args(run_params.storage_config)
    storage_instantiated["artifacts_location_config"] = artifacts_config
    return storage_instantiated


def main():
    """
    Run the evaluation framework pipeline. Evaluation -> Generate_metrics -> visualize -> storage (Optional)
    """

    run_params = get_run_args()

    args, agent_checkpoints = evaluation_args(run_params)

    print("Executing evaluation segment")

    config = eval_load_config(args.cfg)

    config["processors"] = [
        {"type": ConfigSaver},
        {"type": PipelineProcessor, "config": {"run_params": args, "agent_checkpoints": agent_checkpoints}},
    ]

    evaluate_main(args, config)


if __name__ == "__main__":
    main()
