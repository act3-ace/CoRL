"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Functions related to dones for the streamlit app
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.runtime.state.session_state_proxy import SessionStateProxy

from corl.visualization.streamlit_app.utils import convert_counts_to_percentage


def platform_dones_counts_and_table(platform_dones_table: pd.DataFrame, use_container_width: bool) -> None:
    """Generates a streamlit table of platform dones (data in `platform_dones_table`)
    and barcharts grouped by platform name and dones

    Parameters
    ----------
    platform_dones_table : pd.DataFrame
        The platform dones table from EpisodeArtifactTables
    use_container_width : bool
        When True, uses the full container width (streamlit parameter)
    """

    st.markdown("## Platform Dones")
    # Displays the Table of platform dones
    st.dataframe(platform_dones_table, use_container_width=use_container_width)
    _counts_df = (
        platform_dones_table.groupby(["platform_name", "done_name"], as_index=False)
        .agg({"done_triggered": sum})
        .rename(columns={"done_triggered": "count"})
    )
    x_value = "count"
    if st.checkbox("Show as Percentage[%]", key="platform_dones_percentage"):
        _counts_df["percentage"] = convert_counts_to_percentage(input_table=_counts_df, count_column="count")
        x_value = "percentage"
    fig = px.bar(_counts_df, x=x_value, y="done_name", facet_col="platform_name", facet_col_wrap=2)
    st.plotly_chart(fig, use_container_width=use_container_width)


def agent_dones_counts_and_table(
    agent_dones_table: pd.DataFrame, use_container_width: bool, sess_state: SessionStateProxy
) -> SessionStateProxy:
    """Generates a streamlit table of agent dones (data in `agent_dones_table`)
    and barcharts of WLD for a given done condition. The done conditions are selected
    by the user in a drop-down menu.

    Parameters
    ----------
    agent_dones_table : pd.DataFrame
        The agent dones table from EpisodeArtifactTables
    use_container_width : bool
        When True, uses the full container width (streamlit parameter)
    sess_state : SessionStateProxy
        The streamlit session state.

    Returns
    -------
    SessionStateProxy
        The streamlit session state.
    """

    st.markdown("## Agent Dones")
    st.dataframe(agent_dones_table, use_container_width=use_container_width)
    # Get a list of the agent dones
    dones = agent_dones_table["done_name"].unique().tolist()
    dones.sort()
    done_statuses = agent_dones_table["done_status"].unique().tolist()
    # Creates a select box for the done names
    selected_done = st.selectbox(label="Done Name", options=dones)

    # Creates a section of check boxes for the user to select which done
    # statuses (Win, Loss, Draws etc.) should be included
    st.markdown("**Select Done Statuses to Include**")
    done_status_selections = {}
    columns = st.columns(6)
    for idx, col in enumerate(done_statuses):
        with columns[idx]:
            default_value = False
            if col in ["WIN", "PARTIAL_WIN"]:
                default_value = True
            done_status_selections[col] = st.checkbox(label=col, value=default_value)
    include_status = [done_status for done_status, selected in done_status_selections.items() if selected]
    # User option to show as a percentage
    as_percentage = st.checkbox("Show as Percentage[%]")
    if include_status or sess_state.get("agent_done_status_plot_generated"):
        sess_state.agent_done_status_plot_generated = True
        _idx_done_name_filter = agent_dones_table["done_name"] == selected_done
        _idx_selected_statuses = agent_dones_table["done_status"].isin(include_status)
        _subset_df = agent_dones_table.loc[_idx_done_name_filter & _idx_selected_statuses]
        _counts_df = (
            _subset_df.groupby(["agent_id", "done_status"], as_index=False)
            .agg({"test_case": "count"})
            .rename(columns={"test_case": "count"})
        )
        x_value = "count"
        if as_percentage:
            _counts_df["percentage"] = convert_counts_to_percentage(input_table=_counts_df, count_column="count")
            x_value = "percentage"
        fig = px.bar(_counts_df, x=x_value, y="agent_id", color="done_status")
        st.plotly_chart(fig, use_container_width=use_container_width)

    return sess_state
