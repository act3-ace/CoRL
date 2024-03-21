"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Functions for the streamlit episode playback
"""

import plotly.graph_objs as go
import stqdm
import streamlit as st
from streamlit.runtime.state.session_state_proxy import SessionStateProxy

from corl.libraries.functor import Functor
from corl.visualization.plotly_animator import FrameAxisBounds
from corl.visualization.streamlit_app.utils import center_plotly_figure, checkbox_grid, use_container_width


@st.cache_resource
def display_animation(fig: go.Figure, test_case: int):
    """Displays the plotly chart"""
    st.markdown(f"#### Showing Animation for Test Case: {test_case}")
    center_plotly_figure(fig)


def trajectory_playback_section(
    animator: Functor, sess_state: SessionStateProxy, episode_artifacts_dict, frame_axis_bounds: FrameAxisBounds | None = None
) -> SessionStateProxy:
    """Trajectory playback section for streamlit

    Parameters
    ----------
    animator: Functor
        A Functor defining the animator
    sess_state : SessionStateProxy
        The streamlit session state
    episode_artifacts_dict : dict
        A dictionary containing the episode artifacts, where the key is the test
        case number and the value is the episode artifact.
    frame_axis_bounds: FrameAxisBounds, optional
        Sets the default frame axis bounds for the animation.

    Returns
    -------
    SessionStateProxy
        The streamlit session state
    """

    st.markdown("#### Trajectory Animation")
    st.markdown("**The test case selection control is in the `AgentSteps` section**")
    default_x_bounds = frame_axis_bounds.xbounds if frame_axis_bounds else (0, 100)
    default_y_bounds = frame_axis_bounds.ybounds if frame_axis_bounds else (0, 100)
    default_z_bounds = frame_axis_bounds.zbounds if frame_axis_bounds and frame_axis_bounds.zbounds else (0, 100)
    animation_cls = animator.functor(
        episode_artifact=episode_artifacts_dict[sess_state.test_case_selected], progress_bar_func=stqdm.stqdm, **animator.config
    )
    if animation_cls.get_deserializer() is None:
        platform_serializer = str(episode_artifacts_dict[sess_state.test_case_selected].platform_serializer)
        st.warning(f"Skipping Trajectory Animation. No platform deserializer found for {platform_serializer}")
        return sess_state

    # #####################################################################
    # When the frame axis bounds is not provided, display a section
    # for user input for the frame axis bounds.
    # #####################################################################
    st.markdown(
        """
        Set the frame axis bounds for the animation. The animation will
        be played back with these bounds.
        """
    )
    columns = st.columns(6)
    dimensions = animation_cls.get_dimensionality()
    with columns[0]:
        min_x = st.number_input(label="x-axis min", value=default_x_bounds[0])
    with columns[1]:
        max_x = st.number_input(label="x-axis max", value=default_x_bounds[1])
    with columns[2]:
        min_y = st.number_input(label="y-axis min", value=default_y_bounds[0])
    with columns[3]:
        max_y = st.number_input(label="y-axis max", value=default_y_bounds[1])
    frame_axis_obj = FrameAxisBounds(xbounds=(min_x, max_x), ybounds=(min_y, max_y), zbounds=None)

    if dimensions == 3:
        with columns[4]:
            min_z = st.number_input(label="z-axis min", value=default_z_bounds[0])
            max_z = st.number_input(label="z-axis max", value=default_z_bounds[1])
        frame_axis_bounds = FrameAxisBounds(xbounds=(min_x, max_x), ybounds=(min_y, max_y), zbounds=(min_z, max_z))

    # #####################################################################
    # Selection grid for metadata to display above platforms
    # #####################################################################
    possible_selections = animation_cls.get_metadata_names()
    selections = None
    if possible_selections:
        with st.expander(label="**Select Attributes to Display Above Platforms**", expanded=True):
            selections = checkbox_grid(possible_choices=possible_selections, key_prefix="traj_anim")
    animation_button = st.button("Generate Animation")

    # When the test case changes, do not display the animation
    if (
        hasattr(sess_state, "test_case_for_animation")
        and hasattr(sess_state, "test_case_selected")
        and sess_state.test_case_for_animation != sess_state.test_case_selected
    ):
        sess_state.generated_animation = False

    if animation_button:
        animation_cls.set_frame_axis_bounds(frame_axis_obj)
        sess_state.traject_fig, sess_state.traj_stored_frames, sess_state.traj_tables = animation_cls.generate_animation(selections)
        sess_state.generated_animation = True
        sess_state.test_case_for_animation = sess_state.test_case_selected

    if animation_button or sess_state.get("generated_animation"):
        st.markdown("**Note**: Toggle tabs to see underlying raw data in tabular format per platform.")

        platforms = list(sess_state.traj_tables.keys())
        tabs = st.tabs(["Animation"] + [f"Data Table: {platform}" for platform in platforms])
        with tabs[0]:
            display_animation(sess_state.traject_fig, sess_state.test_case_for_animation)
        for idx, platform in enumerate(platforms):
            with tabs[idx + 1]:
                st.markdown(f"##### Underlying Steps Data: {platform}")
                st.dataframe(sess_state.traj_tables[platform], use_container_width=use_container_width)
    return sess_state
