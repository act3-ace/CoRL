"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Default layout for streamlit application
"""
from collections.abc import Callable
from pathlib import Path

import plotly.express as px
import stqdm
import streamlit as st
from pydantic import validate_call

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.recording.folder import FolderRecord
from corl.libraries.functor import Functor
from corl.visualization.episode_artifact_to_tables import EpisodeArtifactTables
from corl.visualization.network_explainability.env_policy_transforms import AgentCheckpoint
from corl.visualization.plotly_animator import FrameAxisBounds
from corl.visualization.streamlit_app.sections.agent_steps import agent_steps_section
from corl.visualization.streamlit_app.sections.platform_and_agent_dones import agent_dones_counts_and_table, platform_dones_counts_and_table
from corl.visualization.streamlit_app.sections.trajectory_animation import trajectory_playback_section
from corl.visualization.streamlit_app.sections.units_conversion import unit_selection_and_table_update
from corl.visualization.streamlit_app.utils import (
    generate_grad_based_feat_importance,
    generate_vf_residuals_dist,
    init_network_explainability_module,
    read_eval_experiment_file,
    st_dataframe_pagination,
    st_directory_picker,
    text_section_policy_network_input,
    use_container_width,
    write_code_block,
)


@st.cache_resource
def episode_artifact_tables(input_path, file_loader):
    """Returns the episode artifact tables"""
    episode_tables = EpisodeArtifactTables(
        input_path=input_path, file_loader=file_loader, status_bar=stqdm.stqdm, print_func=write_code_block
    )
    return episode_tables.tables, episode_tables.evaluation_outcome.episode_artifacts


class DefaultStreamlitPage:
    """Default stremalit app.

    Parameters
    ----------
    file_loader : Callable
        A callable that loads the episode artifacts and test cases. Must return an
        EvaluationOutcome
    """

    def __init__(self, file_loader: Callable = FolderRecord) -> None:
        self.sess_state = st.session_state
        self.file_loader: Callable = file_loader
        self.sess_state.loaded_evaluation = False

        self.tables_dict: dict = {}
        self.episode_artifacts_dict: dict[int, EpisodeArtifact] = {}
        self.obs_units_flattened: dict = {}
        """These are filled out when display_load_episode_artifact_option
        are called"""

        self.full_experiment_path: Path = Path()
        self._root_path: Path = Path()
        self._experiment_dir: str | None = None
        """These are filled out when display_directory_picker is called and a
        valid directory is chosen."""

        self._pages_dict: dict = {}

    def display_directory_picker(self):
        """Displays the directory picker"""
        self._root_path, self._experiment_dir = st_directory_picker(title="Directory Picker")
        if self._experiment_dir is not None and self._root_path.joinpath(self._experiment_dir).exists():
            self.full_experiment_path = self._root_path.joinpath(self._experiment_dir)
            self.sess_state.chosen_experiment_path = True
        else:
            st.warning(
                f""" **Warning: Invalid Selection** - `{self.full_experiment_path}`
            does not contain episode artifacts.
            """
            )

    def display_load_episode_artifact_option(self):
        """Displays the load evaluation button and poopulates the episode artifacts dict
        and tables_dict"""
        if st.button("Load Evaluation"):
            self.sess_state.loaded_evaluation = True
            self.tables_dict, self.episode_artifacts_dict = episode_artifact_tables(self.full_experiment_path, file_loader=self.file_loader)
            self._set_obs_units_dict()

    def _set_obs_units_dict(self):
        _space_table = self.tables_dict["SpaceData"].copy()
        if not _space_table.empty:
            _space_table = _space_table[["attribute_name", "units"]].dropna()
            self.obs_units_flattened = dict(zip(_space_table["attribute_name"], _space_table["units"]))

    def display_config_file(self):
        """Displays the configuration file"""
        if self.sess_state.loaded_evaluation:
            experiment_config, config_file = read_eval_experiment_file(self.full_experiment_path)
            if config_file is not None:
                st.markdown(f"Showing: {self.full_experiment_path.joinpath(config_file)}")
                st.json(experiment_config)

    def display_initial_states_section(self):
        """Displays the initial states table"""
        st.markdown("## Test Cases/ Initial States")
        self._pages_dict = st_dataframe_pagination(tables_dict=self.tables_dict, table_key="InitialStates", pages_dict=self._pages_dict)

    def display_episode_metadata_section(self):
        "Displays the episode metadata section"
        st.markdown("## Episode Metadata")
        self._pages_dict = st_dataframe_pagination(tables_dict=self.tables_dict, table_key="EpisodeMetaData", pages_dict=self._pages_dict)
        fig = px.line(
            data_frame=self.tables_dict["EpisodeMetaData"].sort_values(by=["test_case"]), x="test_case", y="wall_time_sec", markers=True
        )
        st.plotly_chart(fig, use_container_width=use_container_width)

    def display_agent_dones_section(self):
        """Displays the agent done section. When more than one done status is
        selected, shows up as different colored stacked bars"""
        agent_dones_counts_and_table(
            agent_dones_table=self.tables_dict["AgentDoneStatus"], use_container_width=use_container_width, sess_state=self.sess_state
        )

    def display_platform_dones_section(self):
        """Displays the platform done section"""
        platform_dones_counts_and_table(
            platform_dones_table=self.tables_dict["PlatformDoneStates"], use_container_width=use_container_width
        )

    def display_agent_steps_section(self):
        """Displays the agent steps section. A wrapper for `agents_steps_section`"""

        self.sess_state = agent_steps_section(tables_dict=self.tables_dict, sess_state=self.sess_state)

    def display_space_data_table(self):
        """Displays the space data table."""
        space_data_table = self.tables_dict["SpaceData"]
        if not space_data_table.empty:
            st.markdown("## Space Definitions & Units")
            st.dataframe(space_data_table, use_container_width=use_container_width)
            st.markdown(
                """
            **Table**: Provides space (observation and action space) definitions in the original space
            and normalized space."""
            )

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

    def display_policy_network_viz_section(self):
        """Displays the policy network visualization section"""
        st.markdown("# Policy Network Visualizations")
        agent_checkpoints = self._construct_agent_checkpoints()
        env_config_path = self.full_experiment_path.joinpath("env_config.pkl")
        policy_input_map = text_section_policy_network_input(agent_checkpoints=agent_checkpoints)

        clicked_button = st.button("Generate Policy Network Explainability Metrics")

        if clicked_button:
            episode_artifact_files = list(self.full_experiment_path.glob("test_case*/*_episode_artifact.pkl"))

            self.sess_state.eval_net_viz = init_network_explainability_module(
                agent_checkpoints=agent_checkpoints,
                env_config_path=env_config_path,
                policy_network_input_map=policy_input_map,
                episode_artifact_files=episode_artifact_files,
            )
        if self.sess_state.get("eval_net_viz"):
            # #################################################################
            # Displays the Gradient Based Feature Importance
            # #################################################################
            st.markdown("## Policy Network Visualization")
            fig = self.sess_state.eval_net_viz.render_agg_feat_importance_grad_based(normalize=True)
            generate_grad_based_feat_importance(fig)
            # #################################################################
            # Displays the Value Function Residual
            # #################################################################
            fig = self.sess_state.eval_net_viz.render_value_function_residual_dist()
            generate_vf_residuals_dist(fig)

    def _construct_agent_checkpoints(self):
        """Constructs the agent checkpoints list"""
        agent_checkpoints = []
        checkpoints = self.full_experiment_path.joinpath("agent_checkpoints").glob("*")
        for policy_dir in checkpoints:
            agent_name = policy_dir.parts[-1]
            agent_checkpoints.append(AgentCheckpoint(agent_name=agent_name, checkpoint_dir=policy_dir))
        return agent_checkpoints

    def display_units_selector(self):
        """Displays the units selector section"""
        self.tables_dict = unit_selection_and_table_update(tables_dict=self.tables_dict, obs_units_flattened=self.obs_units_flattened)

        # Updates the observation units dictionary
        self._set_obs_units_dict()
