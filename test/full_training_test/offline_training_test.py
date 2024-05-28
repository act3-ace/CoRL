# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

import os
from pathlib import Path

import pytest

from corl.experiments.rllib_utils.episode_artifact_callbacks import PongEpisodeArtifactLogger
from corl.test_utils import full_training
from corl.train_rl import build_experiment, parse_corl_args

rllib_config = {
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
    "evaluation_interval": 1,
    "evaluation_duration": 3,
    "evaluation_num_workers": 1,
    "evaluation_duration_unit": "episodes",
}

output_path = Path(os.path.abspath("./test_generate_episode_artifacts"))


@pytest.mark.dependency()
def test_generate_episode_artifacts(self_managed_ray):
    """Full training test with RLLIB eval enabled. This runs first to generate the episode artifacts
    for the episode artifact reader"""

    args = parse_corl_args(["--cfg", "config/tasks/pong/experiments/pong.yml"])
    experiment_class, experiment_file_validated = build_experiment(args)
    if PongEpisodeArtifactLogger not in experiment_class.config.extra_callbacks:
        experiment_class.config.extra_callbacks.append(PongEpisodeArtifactLogger)

    # os.makedirs(output_path, exist_ok=True)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, rllib_config, output_path)
    experiment_class.run_experiment(experiment_file_validated)

    # #########################################################################
    # Step 2: Check to make sure the evaluation outputs are correct
    # #########################################################################
    experiment_dirs = list(output_path.glob("training/**/eval_checkpoints"))
    assert len(experiment_dirs) > 0, "No experiments found"
    experiment_dir = experiment_dirs[0].parent
    agent_checkpoint_files = list(experiment_dir.joinpath("eval_checkpoints", "epoch_000000", "policies").glob("*"))
    assert len(agent_checkpoint_files) > 0, "No agent checkpoints generated from eval."
    episode_artifact_files = list(experiment_dir.joinpath("trajectories").glob("epoch_000000/*.pickle"))
    assert len(episode_artifact_files) > 0, "No episode artifacts found."


# @pytest.mark.dependency(depends=["test_generate_episode_artifacts"])
def test_episode_artifact_reader(tmp_path, self_managed_ray):
    """Tests the episode artifact reader"""
    # #########################################################################
    # Step 3: Start offline training with MARWIL using the
    # EpisodeArtifactReader
    # #########################################################################
    args = parse_corl_args(["--cfg", "config/tasks/pong/experiments/pong_marwil.yml"])

    experiment_dirs = list(output_path.glob("training/**/eval_checkpoints"))
    experiment_dir = experiment_dirs[0].parent
    offline_config = rllib_config.copy()
    file_directory = os.path.abspath(experiment_dir.joinpath("trajectories/epoch_000000/"))
    offline_config.update(
        {
            "evaluation_config": {"input": "sampler"},
            "input_": "corl.evaluation.runners.section_factories.engine.rllib.offline_rl.episode_artifact_reader.EpisodeArtifactReader",
            "input_config": {"inputs": str(file_directory), "agent_id_in_artifacts": "paddle0_ctrl"},
        }
    )

    experiment_class, experiment_file_validated = build_experiment(args)
    experiment_class.config.tune_config["run_or_experiment"] = "MARWIL"
    if PongEpisodeArtifactLogger in experiment_class.config.extra_callbacks:
        experiment_class.config.extra_callbacks.remove(PongEpisodeArtifactLogger)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, offline_config, tmp_path)
    experiment_class.run_experiment(experiment_file_validated)
