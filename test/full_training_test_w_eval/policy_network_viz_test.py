# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
import pickle
from pathlib import Path

import pytest

from corl.test_utils import full_training
from corl.train_rl import build_experiment, parse_corl_args
from corl.visualization.network_explainability.env_policy_transforms import AgentCheckpoint
from corl.visualization.network_explainability.network_explainability_plotter import PPOTorchNetworkExplainabilityPlotter

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
    "evaluation_interval": 1,
    "evaluation_duration": 3,
    "evaluation_num_workers": 1,
    "evaluation_duration_unit": "episodes",
}


@pytest.mark.parametrize(
    "experiment_config, test_rllib_config",
    [
        pytest.param("config/tasks/docking_1d/experiments/docking_1d.yml", ppo_rllib_config, id="docking-1d"),
        # pytest.param("config/tasks/pong/experiments/pong.yml", ppo_rllib_config, id="pong ppo"),
        # pytest.param("config/tasks/pong/experiments/pong_commander.yml", ppo_rllib_config, id="pong commander ppo"),
    ],
)
@pytest.mark.skip(reason="skip until ray 2.7")
def test_pipeline(experiment_config, test_rllib_config, tmp_path, self_managed_ray):
    """Full training test with RLLIB eval enabled and generates the policy network
    visualizations, without rendering the figure."""

    args = parse_corl_args(["--cfg", experiment_config])
    experiment_class, experiment_file_validated = build_experiment(args)
    full_training.update_rllib_experiment_for_test(experiment_class, experiment_file_validated, test_rllib_config, tmp_path)
    experiment_class.run_experiment(experiment_file_validated)

    # #########################################################################
    # Step 2: Check to make sure the evaluation outputs are correct
    # #########################################################################
    experiment_dirs = list(tmp_path.glob("training/**/eval_checkpoints"))
    assert len(experiment_dirs) > 0, "No experiments found"
    experiment_dir = experiment_dirs[0].parent
    agent_checkpoint_files = list(experiment_dir.joinpath("eval_checkpoints", "epoch_000000", "policies").glob("*"))
    assert len(agent_checkpoint_files) > 0, "No agent checkpoints generated from eval."
    params_file = experiment_dir.joinpath("params.pkl")
    episode_artifact_files = list(experiment_dir.joinpath("trajectories").glob("epoch_000000/*.pickle"))
    assert len(episode_artifact_files) > 0, "No episode artifacts found."

    # #########################################################################
    # Step 3: Generate the policy network visualizations
    # (without showing the figure)
    # #########################################################################
    # Setup Inputs
    agent_checkpoints = []
    for checkpoint_dir in agent_checkpoint_files:
        agent_name = Path(checkpoint_dir).name
        agent_checkpoints.append(AgentCheckpoint(agent_name=agent_name, checkpoint_dir=checkpoint_dir))

    with open(params_file, "rb") as file_obj:
        params = pickle.load(file_obj)  # noqa: S301

    explainer_viz = PPOTorchNetworkExplainabilityPlotter(agent_checkpoints=agent_checkpoints, env_config=params["env_config"])

    for filename in episode_artifact_files:
        with open(filename, "rb") as file_obj:
            episode_artifact = pickle.load(file_obj)  # noqa: S301
        explainer_viz.append_feat_importance_gradient_based(episode_artifact)
        explainer_viz.append_vf_residuals(episode_artifact=episode_artifact)

    # Render the distributions
    explainer_viz.render_value_function_residual_dist()
    explainer_viz.render_agg_feat_importance_grad_based()
    # Call fig.show() to see figures
