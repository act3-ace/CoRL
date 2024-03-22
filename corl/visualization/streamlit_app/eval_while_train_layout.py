"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Layout for visualizing evaluations generated from training.
"""
import pickle
from collections.abc import Callable

import streamlit as st
from pydantic import validate_call

from corl.evaluation.recording.folder import FolderRecord
from corl.evaluation.runners.section_factories.engine.rllib.episode_artifact_logging_callback import EVAL_CHECKPOINTS_SUBDIR
from corl.libraries.functor import Functor
from corl.visualization.network_explainability.env_policy_transforms import AgentCheckpoint
from corl.visualization.plotly_animator import FrameAxisBounds
from corl.visualization.streamlit_app.default_layout import DefaultStreamlitPage, episode_artifact_tables
from corl.visualization.streamlit_app.sections.trajectory_animation import trajectory_playback_section
from corl.visualization.streamlit_app.utils import (
    _check_required_subdirectories_policy_viz,
    _group_checkpoints_by_input_output,
    create_selection_grid_policy_is_agent,
    create_selection_grid_single_and_self_policy,
    generate_grad_based_feat_importance,
    generate_vf_residuals_dist,
    init_network_explainability_module,
    st_directory_picker,
    text_section_policy_network_input,
)


class StreamlitPageEvaluationWhileTraining(DefaultStreamlitPage):
    """Streamlit page specific to evaluation runs generated from training"""

    def __init__(self, file_loader: Callable = FolderRecord) -> None:
        super().__init__(file_loader)
        self.epoch_selection: str = ""

    def display_directory_picker(self):
        """Displays the directory picker"""
        self._root_path, self._experiment_dir = st_directory_picker(title="Directory Picker")
        if self._experiment_dir is not None and self._root_path.joinpath(self._experiment_dir, "trajectories").exists():
            self.full_experiment_path = self._root_path.joinpath(self._experiment_dir)
            self.sess_state.chosen_experiment_path = True
        else:
            st.warning(
                f""" **Warning: Invalid Selection** - `{self.full_experiment_path}`
            does not contain episode artifacts. The episode
            artifacts are expected in a subdirectory named "trajectories".
            """
            )

    def display_epoch_picker(self):
        """Displays the epoch picker: this is specific to evaluation runs generated from training"""
        # This prevents a reload of the trajectory list while its still being populated
        if self.sess_state.get("chosen_experiment_path"):
            eval_dir = self.full_experiment_path.joinpath("trajectories/")
            self.sess_state.epoch_names = [f.name for f in eval_dir.iterdir()]
            self.sess_state.epoch_names.sort()
            self.sess_state.already_loaded_epochs_dir = True
            epoch_selection = st.selectbox("Pick an Epoch to Inspect:", self.sess_state.epoch_names, key="epoch_")
            if epoch_selection is not None:
                self.epoch_selection = epoch_selection

    def display_load_episode_artifact_option(self):
        """Displays the load evaluation button and poopulates the episode artifacts dict
        and tables_dict"""
        if st.button("Load Evaluation"):
            self.sess_state.loaded_evaluation = True
            self.tables_dict, self.episode_artifacts_dict = episode_artifact_tables(
                self.full_experiment_path.joinpath("trajectories", self.epoch_selection), file_loader=self.file_loader
            )
            self._set_obs_units_dict()

    def _construct_agent_checkpoints(self, use_placeholder=False):
        """Creates a list of agent checkpoints."""

        agent_checkpoints = []
        # Use the epoch selection if it exists, otherwise create one
        epoch_selection = self.epoch_selection
        if use_placeholder:
            epochs = list(self.full_experiment_path.joinpath(EVAL_CHECKPOINTS_SUBDIR).glob("epoch*"))
            epochs.sort()
            epoch_selection = epochs[0]

        checkpoints = self.full_experiment_path.joinpath(EVAL_CHECKPOINTS_SUBDIR, epoch_selection, "policies").glob("*")
        for policy_dir in checkpoints:
            agent_name = policy_dir.parts[-1]
            agent_checkpoints.append(AgentCheckpoint(agent_name=agent_name, checkpoint_dir=policy_dir))
        return agent_checkpoints

    @validate_call
    def display_trajectory_animation_section(self, animator: Functor, frame_axis_bounds: FrameAxisBounds | None = None):
        """
        Displays the Trajectory Animation Section

        Parameters
        ----------
        animator: Functor
            A Functor defining the animator
        frame_axis_bounds: FrameAxisBounds, optional
            Sets the default frame axis bounds for the animation.
        """
        self.sess_state = trajectory_playback_section(
            animator=animator,
            sess_state=self.sess_state,
            episode_artifacts_dict=self.episode_artifacts_dict,
            frame_axis_bounds=frame_axis_bounds,
        )

    def display_policy_network_viz_section(self) -> None:  # noqa: PLR0915
        # #########################################################################
        # Policy Network Visualization
        st.markdown("## Policy Network Visualization")
        st.markdown(f"**Training Run**: {self.full_experiment_path}")
        st.markdown(
            """
            ### Note
            - The compute time scales linearly with the number of checkpoints.
            - When running Policy Network Visualization for an active training run, make sure
            all trajectories have been generated for the epoch chosen, otherwise the results will change
            as more trajectories are generated.
            """
        )
        _check_required_subdirectories_policy_viz(self.full_experiment_path)

        # Use placeholder - we just want to grab the names of the agents at this point.
        agent_checkpoints = self._construct_agent_checkpoints(use_placeholder=True)
        policy_input_map = text_section_policy_network_input(agent_checkpoints=agent_checkpoints)
        columns = st.columns(3)
        single_policy_or_self_play = "Single-Policy / Self-Play"
        policy_is_agent = "PolicyIsAgent"
        clicked = False
        # #########################################################################
        # Select: Number of checkpoints to examine
        with columns[0]:
            num_checkpoints = st.number_input(label="Number of Checkpoints to Examine", value=1, min_value=1, step=1)
        with columns[1]:
            policy_map = st.radio(label="Policy Map", options=[single_policy_or_self_play, policy_is_agent], horizontal=True)
        # #########################################################################
        # Selected: SinglePolicy and Self-Play force a main-policy
        if policy_map == single_policy_or_self_play:
            with columns[2]:
                main_policy_name = st.selectbox(label="Main Policy Name", options=list(policy_input_map.keys()))
            num_selections_clicked = st.button("Generate Selections")
            if num_selections_clicked or hasattr(self.sess_state, "generated_single_policy_selfplay_selections"):
                agent_checkpoint_selections = create_selection_grid_single_and_self_policy(
                    num_checkpoints, self.full_experiment_path, main_policy=main_policy_name
                )
                clicked = st.button("Done")
                if clicked:
                    self.sess_state.agent_checkpoints = agent_checkpoint_selections
                    self.sess_state.generated_policy_viz_eval_train = False
                self.sess_state.generated_single_policy_selfplay_selections = True

        # #########################################################################
        # Selected: PolicyIsAgent allows the user to select both an epoch and
        # policy to review
        if policy_map == policy_is_agent:
            num_selections_clicked = st.button("Generate Selections")
            if num_selections_clicked or hasattr(self.sess_state, "generated_policy_is_agent_selections"):
                agent_checkpoint_selections = create_selection_grid_policy_is_agent(num_checkpoints, self.full_experiment_path)

                clicked = st.button("Done")
                if clicked:
                    self.sess_state.agent_checkpoints = agent_checkpoint_selections
                    self.sess_state.generated_policy_viz_eval_train = False

                self.sess_state.generated_policy_is_agent_selections = True

        # ####################################################################
        # Generates the Features: Only run when a selection is made and
        # visualizations have not been generated
        if clicked and self.sess_state.get("agent_checkpoints") and not self.sess_state.get("generated_policy_vz_eval_train", False):
            # Loads the environment configuration from params.pkl
            with open(self.full_experiment_path.joinpath("params.pkl"), "rb") as file_obj:
                params = pickle.load(file_obj)  # noqa: S301

            # Dictionary to hold the explainability module for each selection made
            # where the key is the shortened checkpoint path
            explain_modules: dict[str, dict] = {"explainer_class": {}, "vf_residuals": {}}
            # A list of dictionaries where each dictionary stores the
            # checkpoint path, policy inputs and policy outputs
            input_outputs = []
            explain_cls = None
            # Iterate over selections
            for agent_selections_dict in self.sess_state.agent_checkpoints:
                # Grab the agent checkpoint and the associated set of trajectories for the selected epoch
                agent_checkpoint = agent_selections_dict["agent_checkpoint"]
                episode_artifact_files = list(
                    self.full_experiment_path.joinpath("trajectories", agent_selections_dict["epoch_selected"]).glob("*.pickle")
                )

                # Initializes the explainability module
                explain_cls = init_network_explainability_module(
                    agent_checkpoints=[agent_checkpoint],
                    env_config=params["env_config"],
                    verbose=False,
                    policy_network_input_map=policy_input_map,
                    episode_artifact_files=episode_artifact_files,
                )

                checkpoint_path = str(agent_checkpoint.checkpoint_dir)
                prefix = str(self.full_experiment_path.joinpath(EVAL_CHECKPOINTS_SUBDIR))

                checkpoint_path = checkpoint_path.replace(prefix, "")

                # Store visualization outputs
                explain_modules["explainer_class"].update(
                    {checkpoint_path: {"class": explain_cls, "agent_id": agent_checkpoint.agent_name}}
                )
                explain_modules["vf_residuals"].update({checkpoint_path: explain_cls.vf_residuals})

                net_input_names = explain_cls.network_input_names.copy()
                net_output_names = explain_cls.network_output_names.copy()
                net_input_names.sort()
                net_output_names.sort()
                policy_input_string = ".".join(net_input_names)
                policy_output_string = ".".join(net_output_names)
                input_outputs.append({"checkpoint": checkpoint_path, "input": policy_input_string, "output": policy_output_string})

            # Group by policy networks with the same input and outputs
            if explain_cls is not None:
                groups = _group_checkpoints_by_input_output(input_outputs)
                self.sess_state.feat_importance_figs = []
                for group in groups:
                    grouped_feats = {}
                    for checkpoint_key in group:
                        explain_cls = explain_modules["explainer_class"][checkpoint_key]["class"]
                        agent_name = explain_modules["explainer_class"][checkpoint_key]["agent_id"]
                        assert explain_cls is not None
                        feats_dict = explain_cls.get_agg_feat_importance_grad_based(agent_id=agent_name)
                        grouped_feats[checkpoint_key] = feats_dict[agent_name]
                    assert explain_cls is not None
                    self.sess_state.feat_importance_figs.append(
                        {"feat_importance_grad_based": explain_cls.render_agg_feat_importance_grad_based(feats_dict=grouped_feats)}
                    )

                # iterate through the value function residuals and change the agent identifier to the checkpoint path
                combined_vf_residuals = {}
                for checkpoint_path, vf_residuals_dict in explain_modules["vf_residuals"].items():
                    assert len(vf_residuals_dict) == 1
                    for vf_residuals in vf_residuals_dict.values():
                        combined_vf_residuals[checkpoint_path] = vf_residuals
                    # Since we pass in the vf_residuals, we can use any instance of explain_cls
                    assert explain_cls is not None
                    fig = explain_cls.render_value_function_residual_dist(vf_residuals=combined_vf_residuals)
                    self.sess_state.residual_dist = fig
                # Flip the flag so that it does not rerun
                self.sess_state.generated_policy_viz_eval_train = True

        # When feat_importance_figs is defined, plot the figures for each group
        if hasattr(self.sess_state, "feat_importance_figs"):
            for fig_dict in self.sess_state.feat_importance_figs:
                generate_grad_based_feat_importance(fig_dict["feat_importance_grad_based"])
        if hasattr(self.sess_state, "residual_dist"):
            generate_vf_residuals_dist(self.sess_state.residual_dist)
