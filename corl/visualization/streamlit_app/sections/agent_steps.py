"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Functions for agent steps section of the stremalit app.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from streamlit.runtime.state.session_state_proxy import SessionStateProxy

from corl.visualization.episode_artifact_to_tables import FlattenedAgentStep
from corl.visualization.streamlit_app.utils import checkbox_grid, confidence_interval, st_dataframe_pagination, ui_dataframe_filter


def agent_steps_section(  # noqa: PLR0915
    tables_dict: dict[str, pd.DataFrame],
    sess_state: SessionStateProxy,
) -> SessionStateProxy:
    """
    Creates the streamlit Agent Steps section

    Parameters
    ----------
    tables_dict : Dict[str, pd.DataFrame]
        This is expected to be EpisodeArtifactTables.tables
    sess_state : SessionStateProxy
        The streamlit session state
    obs_units_flattened : Dict[str, str]
        A dictionary where the key corresponds to the attribute name and the
        value corresponds to the unit (str)
    """

    _section_headers()
    sess_state.show_mean_with_ci = False
    agent_step_table = tables_dict["FlattenedAgentStep"]

    ###########################################################################
    # User Input: Select Attributes to Plot
    ###########################################################################
    user_selected_attributes = expander_select_attributes(agent_step_table)

    test_case_idx = st.number_input(
        label="Test Case Number", min_value=agent_step_table["test_case"].min(), max_value=agent_step_table["test_case"].max()
    )

    sess_state.test_case_selected = test_case_idx
    attributes_filter = agent_step_table["attribute_name"].isin(user_selected_attributes)
    # Depending on the selections below, the test case filter may not be used.
    agents = agent_step_table["agent_id"].unique().tolist()
    agents.sort()

    # #####################################################
    # User selections for how to display plots: Radio
    # buttons
    # #####################################################
    # Options for displaying on the same / separate plots
    # and options to show all trajectories (raw) and by
    # aggregate.
    options_dict = {
        "same_plot": "Display traces on the same plot",
        "different_plots": "Display traces on separate plots",
        "show_all": "Show all test cases",
        "aggregate": "Aggregate across test cases (mean +/- 95% CI)",
    }
    plot_layout = (
        st.radio(label="radio", label_visibility="hidden", horizontal=True, options=list(options_dict.values()))
        or options_dict["same_plot"]
    )

    sess_state.show_mean_with_ci = plot_layout == options_dict["aggregate"]
    if not hasattr(sess_state, "custom_y_axes_dict"):
        sess_state.custom_y_axes_dict = {}
        sess_state.custom_y_axes_dict = {_agent_id: {} for _agent_id in agents}

    # #####################################################################
    # Displays different agents in different tabs
    # #####################################################################
    agent_tabs = st.tabs(agents)
    for idx, agent_id in enumerate(agents):
        # Filter by agent and selected attributes
        agent_filter = agent_step_table["agent_id"] == agent_id
        # Makes a copy so that we do not modify the underlying dataframe
        _subset_df = agent_step_table.loc[attributes_filter & agent_filter].copy()
        if _subset_df.empty:
            continue
        # Creates agent tabs
        with agent_tabs[idx]:
            # #####################################################################
            # Displays Plots & Data and nested tab for each agent
            # #####################################################################
            plot_and_data_tabs = st.tabs(["Plots", "Tabular Data"])
            # #################################################################
            # Nested Tab 0: Plotly Figure & Options
            # #################################################################
            with plot_and_data_tabs[0]:
                # #############################################################
                # User Selections for how to display plots: Checkbox options
                # #############################################################
                _subset_df, plot_kwargs = _expander_checkboxes_xaxes_options_and_filter(_subset_df)

                attributes_in_dataframe = _subset_df["attribute_name"].unique().tolist()
                attributes_in_dataframe.sort()

                # #############################################################
                # Y-axis View Options
                # #############################################################
                _expander_set_yaxes_view_options(
                    agent_steps_table=_subset_df,
                    agent_id=agent_id,
                    plot_layout=plot_layout,
                    options_dict=options_dict,
                    sess_state=sess_state,
                    attributes_in_dataframe=attributes_in_dataframe,
                )

                # #############################################################
                # Filters dataframe when necessary & builds the plotly kwargs
                # #############################################################
                test_case_filter = _subset_df["test_case"] == test_case_idx

                # Aggregate with mean and 95% CI option
                # When shown by aggregate, each attributed selected must be on a separate plot
                # for the 95% confidence intervals to render correctly.
                if plot_layout == options_dict["aggregate"]:
                    _subset_df = _expander_filter_dataframe(tables_dict=tables_dict, filtered_agent_table=_subset_df, sess_state=sess_state)
                    _subset_df, plot_kwargs = _plot_layout_aggregate_option(agent_steps_table=_subset_df, plot_kwargs=plot_kwargs)
                    sess_state.show_mean_with_ci = True

                # Display on separate plots
                if plot_layout == options_dict["different_plots"]:
                    plot_kwargs = _plot_layout_different_plots_option(agent_steps_table=_subset_df, plot_kwargs=plot_kwargs)
                    _subset_df = _subset_df.loc[test_case_filter]

                # Display on the same plot
                if plot_layout == options_dict["same_plot"]:
                    plot_kwargs = _plot_layout_same_plot_option(plot_kwargs=plot_kwargs)
                    _subset_df = _subset_df.loc[test_case_filter]

                # Display all test cases: when all traces across test cases are shown,
                # each attribute gets its own plot
                if plot_layout == options_dict["show_all"]:
                    _subset_df = _expander_filter_dataframe(tables_dict=tables_dict, filtered_agent_table=_subset_df, sess_state=sess_state)
                    plot_kwargs = _plot_layout_show_all_option(agent_steps_table=_subset_df, plot_kwargs=plot_kwargs)

                # #####################################################################
                # Creates the Plotly Figure
                # #####################################################################
                category_orders = {"attribute_name": attributes_in_dataframe}
                # When the test case column is present, order by test case
                if "test_case" in _subset_df.columns:
                    test_cases = _subset_df["test_case"].unique().tolist()
                    test_cases.sort()
                    category_orders.update({"test_case": test_cases})

                fig = px.line(_subset_df, markers=True, category_orders=category_orders, hover_data=_subset_df.columns, **plot_kwargs)
                if sess_state.get("show_mean_with_ci"):
                    fig = update_figure_with_confidence_intervals(agent_steps_table=_subset_df, fig=fig, plot_kwargs=plot_kwargs)

                if hasattr(sess_state, "custom_y_axes_dict") and sess_state.custom_y_axes_dict[agent_id]:
                    fig = update_figure_with_y_axes_view_options(fig=fig, custom_y_axes_dict=sess_state.custom_y_axes_dict[agent_id])

                # #############################################################
                # Display the Figure on streamlit
                # #############################################################
                # Do not sync the y-axes for faceted plots
                fig.update_yaxes(matches=None)
                fig.update_yaxes({"tickfont": {"size": 20}})
                fig.update_xaxes({"tickfont": {"size": 20}, "titlefont": {"size": 20}})
                st.plotly_chart(fig, use_container_width=True)

            # #################################################################
            # Nested Tab 2: Tabular data for underlying Plotly Figure
            # #################################################################
            with plot_and_data_tabs[1]:
                # Save the state of the page so that it doesn't restart upon every call
                sess_state.pages_dict = st_dataframe_pagination(
                    dataframe=_subset_df, table_key="FlattenedAgentStep", pages_dict=sess_state.get("pages_dict", {})
                )
                st.markdown(
                    f"""**Table**: Displaying a subset of the agents steps table.
                    The field `attribute_name` is limited to: ```{attributes_in_dataframe}```"""
                )
    return sess_state


def expander_select_attributes(agent_step_table: pd.DataFrame) -> list[str]:
    """Creates a section grouped by attribute type (e.g. observations, actions, rewards) for a grid
    of attributes to select from. Returns the selected attributes.

    Parameters
    ----------
    agent_step_table : pd.DataFrame
        The agent steps table from EpisodeArtifactTable
    obs_units_flattened : dict
        A dictionary where the key corresponds to the attribute name (str)
        and the value corresponds to the units (str)

    Returns
    -------
    List[str]
        A list of selected attributes
    """
    selections_table = agent_step_table[["attribute_descrip", "attribute_name", "units"]].drop_duplicates()
    idx_has_units = selections_table["units"].notnull()
    selections_table["attribute_name_w_units"] = selections_table["attribute_name"].copy()
    selections_table.loc[idx_has_units, "attribute_name_w_units"] = (
        selections_table.loc[idx_has_units, "attribute_name"] + "[" + selections_table.loc[idx_has_units, "units"] + "]"
    )
    user_selected_attributes = []

    with st.expander("**Select Observations to Plot**", expanded=True):
        st.markdown("**Observations**")
        possible_selections = selections_table.loc[
            selections_table["attribute_descrip"] == FlattenedAgentStep.AttributeDescription.observation
        ]["attribute_name"].tolist()

        possible_selections_with_units = selections_table.loc[
            selections_table["attribute_descrip"] == FlattenedAgentStep.AttributeDescription.observation
        ]["attribute_name_w_units"].tolist()

        # # NOTE: We append units to the display of possible units, but this is not how they are stored within the table.
        user_selected_attributes.extend(
            checkbox_grid(
                possible_choices=possible_selections_with_units,
                key_prefix="obs_corl_eval",
                possible_choices_return_value=possible_selections,
            )
        )

    with st.expander("**Select Actions to Plot**", expanded=True):
        st.markdown("**Actions**")
        possible_selections = selections_table.loc[selections_table["attribute_descrip"] == FlattenedAgentStep.AttributeDescription.action][
            "attribute_name"
        ].tolist()
        # NOTE: We append units to the display of possible units;
        # this is not how they are stored within the table so we need to keep
        # track of it separately.
        possible_selections_with_units = selections_table.loc[
            selections_table["attribute_descrip"] == FlattenedAgentStep.AttributeDescription.action
        ]["attribute_name_w_units"].tolist()

        user_selected_attributes.extend(
            checkbox_grid(
                possible_choices=possible_selections_with_units,
                key_prefix="action_corl_eval",
                possible_choices_return_value=possible_selections,
            )
        )

    with st.expander("**Select Rewards to Plot**", expanded=True):
        st.markdown("**Rewards**")
        possible_selections = selections_table.loc[
            selections_table["attribute_descrip"].isin(
                [FlattenedAgentStep.AttributeDescription.reward, FlattenedAgentStep.AttributeDescription.cumulative_reward]
            )
        ]["attribute_name"].tolist()
        user_selected_attributes.extend(checkbox_grid(possible_choices=possible_selections, key_prefix="reward_corl_eval"))

    return user_selected_attributes


def append_confidence_interval_columns(
    dataframe: pd.DataFrame,
    steps_col: str = "steps",
    attribute_name_col: str = "attribute_name",
    attribute_value_col: str = "attribute_value",
    conf_interval: int = 95,
):
    """Creates two columns indicating the upper and lower bounds of the
    confidence interval, where which confidnece interval is defined by
    `conf_interval`.

    Groups the `dataframe` by `steps_col` and `attribute_name` and applies
    the function `utils.confidence_interval` to compute the confidence
    intervals.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A dataframe with at least the FlattenedAgentSteps schema.
        That is, this should be a trajectory.
    steps_col : str, optional
        The column name where the steps are stored, by default 'steps'
    attribute_name_col : str, optional
        The name of the column that stores the attribute names, by default 'attribute_name'
    attribute_value_col : str, optional
        The name of the column that stores the attribute values, by default 'attribute_value'
    conf_interval : int, optional
        An number indicating the confidence interval, by default 95.

    Returns
    -------
    pd.DataFrame:
        A dataframe with columns: `steps_col`, `attribute_name_col`,
        `<attribute_value_col>_mean`, `<attribute_value_col>_ci_lower`, `<attribute_value_col>_ci_upper`
    """

    dataframe = dataframe.copy()
    dataframe[f"{attribute_value_col}_mean"] = dataframe[attribute_value_col].copy()
    dataframe[f"{attribute_value_col}_ci_lower"] = dataframe[attribute_value_col].copy()
    dataframe[f"{attribute_value_col}_ci_upper"] = dataframe[attribute_value_col].copy()
    return dataframe.groupby([steps_col, attribute_name_col], as_index=False).agg(
        {
            f"{attribute_value_col}_mean": "mean",
            f"{attribute_value_col}_ci_lower": lambda column_values: confidence_interval(column_values, which=conf_interval)[0],
            f"{attribute_value_col}_ci_upper": lambda column_values: confidence_interval(column_values, which=conf_interval)[1],
        }
    )


def update_figure_with_confidence_intervals(agent_steps_table: pd.DataFrame, fig: go.Figure, plot_kwargs: dict) -> go.Figure:
    """Updates the plotly figure with the custom 95% confidence interval view

    Parameters
    ----------
    agent_steps_table : pd.DataFrame
        The agent steps table from EpisodeArtifactTable or a subset of it.
    fig : go.Figure
        The plotly figure object with the mean already plotted.
    plot_kwargs : Dict
        plotly kwargs so far.

    Returns
    -------
    go.Figure
        Updated figure.
    """

    # The order the plots appear may not match how its internally stored and referenced by
    # the column / row within plotly. We grab the layout object and check the name of the title
    # to ensure the 95% CI are plotted on the correct facet.
    for idx, _layout in enumerate(fig.layout.annotations):
        _attribute_name = _layout.text.replace("attribute_name=", "")
        _subset_by_attribute = agent_steps_table.loc[agent_steps_table["attribute_name"] == _attribute_name]
        fig.add_trace(
            go.Scatter(
                name="CI Upper",
                x=_subset_by_attribute[plot_kwargs["x"]],
                y=_subset_by_attribute["attribute_value_ci_upper"],
                line={"width": 0},
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                showlegend=False,
                marker={"color": "#444"},
            ),
            col=1,
            row=idx + 1,
        )
        # Note: we "fill to next y". This will not work properly
        # when we plot more than one trace on a single figure. As such,
        # we limit the view of aggregated traces with the mean and CI
        # to one attribute per plot.
        fig.add_trace(
            go.Scatter(
                name="CI Lower",
                x=_subset_by_attribute[plot_kwargs["x"]],
                y=_subset_by_attribute["attribute_value_ci_lower"],
                line={"width": 0},
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=False,
                marker={"color": "#444"},
            ),
            col=1,
            row=idx + 1,
        )
    return fig


def update_figure_with_y_axes_view_options(fig: go.Figure, custom_y_axes_dict: dict) -> go.Figure:
    """Updates the figure with the y-axes selected and stored in the `sess_state.custom_y_axes_dict`

    Parameters
    ----------
    fig : go.Figure
        Existing plotly figure
    custom_y_axes_dict : Dict
        A dictionary where the key is the attribute name and the value is the y-axis prop

    Returns
    -------
    go.Figure
        Updated plotly figure
    """
    if len(custom_y_axes_dict) > 1:
        _yaxis_prop = {}
        for idx, _layout in enumerate(fig.layout.annotations):
            y_axis_name = "yaxis" if idx == 0 else f"yaxis{idx+1}"
            # Grab the y-axis properties according to the attribute name
            _prop = custom_y_axes_dict.get(_layout.text.replace("attribute_name=", ""))
            if _prop:
                _yaxis_prop[y_axis_name] = _prop

        fig.update_layout(**_yaxis_prop)
    else:
        _yaxis_prop = {"yaxis": custom_y_axes_dict[""]}
        fig.update_layout(**_yaxis_prop)

    return fig


def _expander_checkboxes_xaxes_options_and_filter(agent_steps_table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Section for x-axis options (1) Display Relavtive to last step
    and (2) Filter on Steps.

    Parameters
    ----------
    agent_steps_table : pd.DataFrame
        The agent steps table from EpisodeArtifactTable or a subset of it.
    pd.DataFrame
        The potentially modified agent_steps_table
    Dict
        Initial plotly kwargs
    """

    plot_kwargs: dict = {}
    # #########################################################################
    # User selections for how to display plots: Use Expander to match Y-axis
    # options
    # #########################################################################
    hash_id = "_".join(agent_steps_table["agent_id"].unique().tolist())
    with st.expander("**X-Axis View Options**", expanded=True):
        # X-axis: Time or Steps?
        # time = steps / frame_rate, where frame_rate is an optional value
        # when the frame_rate is None, the time columns will be null
        if agent_steps_table["time"].isnull().sum() == 0:
            step_time_option = (
                st.radio(label="Axis Index", options=["step", "time"], horizontal=True, key=f"{hash_id}_xaxis_index") or "step"
            )

        columns = st.columns(4)
        # Changes the x-axis to be relative to last step / second. That
        # is, uses the `step_rel_to_end` index rather than `step.
        with columns[0]:
            if st.checkbox("Display relative to last step / second", key=f"{hash_id}_rel_to_end"):
                plot_kwargs.update({"x": f"{step_time_option}_rel_to_end"})
            else:
                plot_kwargs.update({"x": step_time_option})
        # Filters by step: allows the user to input a min / max
        # for the step range.
        with columns[1]:
            if st.checkbox("Filter on x-axis index", key=f"{hash_id}_filter_on_steps"):
                with columns[2]:
                    min_value = agent_steps_table[plot_kwargs["x"]].min()
                    max_value = agent_steps_table[plot_kwargs["x"]].max()

                    selected_min = st.number_input("start", min_value=min_value, max_value=max_value, key=f"{hash_id}_start")
                with columns[3]:
                    selected_max = st.number_input("end", min_value=min_value, max_value=max_value, value=max_value, key=f"{hash_id}_end")
                steps_filter = agent_steps_table[plot_kwargs["x"]].between(selected_min, selected_max, inclusive="both")
                agent_steps_table = agent_steps_table.loc[steps_filter]

    return agent_steps_table, plot_kwargs


def _expander_filter_dataframe(tables_dict: dict, filtered_agent_table: pd.DataFrame, sess_state: SessionStateProxy) -> pd.DataFrame:
    """
    Dataframe filter GUI
    """
    table_names = [
        name for name, pd_table in tables_dict.items() if "test_case" in pd_table.columns and name != FlattenedAgentStep.__name__
    ]
    _agent_id_key = filtered_agent_table["agent_id"].unique()[0]
    with st.expander("**Advanced: Filter Test Cases (Show All / Aggregate Only)**"):
        st.markdown(
            """ **Overview**: This section allows the user to filter down which test cases to show
                  when all the test cases are being displayed.
                  First choose which tables to filter on and press `DONE`.
                  The filters will stack together as `AND` operations.
                  A view of the filtered data for each table is shown for reference.
                  Press `Clear Filters` to clear all filters. """
        )
        if st.button("Clear Filters", key=f"clear_filter_{_agent_id_key}"):
            sess_state.tables_selected_for_filter = False
        with st.form(f"table_selections_{_agent_id_key}"):
            selected_tables = st.multiselect(
                label="Choose Which Tables to Filter on", options=table_names, key=f"multi_select_{_agent_id_key}"
            )
            tables_selected_button = st.form_submit_button("Done")
        if tables_selected_button or sess_state.get("tables_selected_for_filter"):
            # Start with all test cases in agent table
            filtered_test_cases = set(filtered_agent_table["test_case"].unique())
            for table_name in selected_tables:
                _dataframe = tables_dict[table_name].copy()
                _dataframe = ui_dataframe_filter(_dataframe, markdown_title=f"### Table: {table_name}")
                filtered_test_cases = filtered_test_cases.intersection(set(_dataframe["test_case"].unique()))
                sess_state.tables_selected_for_filter = True

            filtered_test_cases_as_list = list(filtered_test_cases)
            filtered_test_cases_as_list.sort()
            st.markdown(f"**Filtered Test Cases**: `{filtered_test_cases_as_list}`")

            idx = filtered_agent_table["test_case"].isin(filtered_test_cases)
            return filtered_agent_table.loc[idx]

    return filtered_agent_table


def _expander_set_yaxes_view_options(
    agent_steps_table: pd.DataFrame,
    agent_id: str,
    plot_layout: str,
    options_dict: dict[str, str],
    sess_state: SessionStateProxy,
    attributes_in_dataframe: list[str],
) -> SessionStateProxy:
    """Expander section to set y-axis bounds for each plot.

    Parameters
    ----------
    agent_steps_table : pd.DataFrame
        The agent steps table from EpisodeArtifactTable or a subset of it.
    plot_layout : str
        The radio selection output, this is a value of `options_dict`
    options_dict : Dict[str, str]
        The dictionary of options for how to display the plots
    sess_state : SessionStateProxy
        Streamlit session state
    attributes_in_dataframe : List[str]
        The selected attributes that are present in the dataframe, ordered.
        The order of this list will dictate the order of the y-axis attributes
        shown in this section.

    Returns
    -------
    SessionStateProxy
        Streamlit session state.
    """

    hash_id = "_".join(agent_steps_table["agent_id"].unique().tolist())
    with st.expander(label="**Y-Axis View Options**", expanded=False):
        # Option to show magnitude of y-axis values only.
        apply_abs_value = st.checkbox(label="Show Absolute Values", key=f"mag_{hash_id}")
        if apply_abs_value:
            agent_steps_table["attribute_value"] = agent_steps_table["attribute_value"].abs()
        st.markdown("Defaults to the min & max values (with a 5% buffer) across test-cases for each attribute.")
        with st.form(f"yaxis_options_{hash_id}"):
            # Only one y-axis when same plot is selected.
            iterate_over = [""] if plot_layout == options_dict["same_plot"] else attributes_in_dataframe
            _output_dict = {}
            default_min_y = agent_steps_table["attribute_value"].min() * 0.95
            default_max_y = agent_steps_table["attribute_value"].max() * 1.05
            for _attribute in iterate_over:
                st.markdown(f"{_attribute}")
                # When plotted on separate plots, min / max defaults are specific to each selected attributed
                if _attribute in attributes_in_dataframe:
                    default_min_y = agent_steps_table.loc[agent_steps_table["attribute_name"] == _attribute]["attribute_value"].min() * 0.95
                    default_max_y = agent_steps_table.loc[agent_steps_table["attribute_name"] == _attribute]["attribute_value"].max() * 1.05
                columns = st.columns(2)
                # NOTE: -0.05 / 1.05 when min / max values evaluate to 0 to create a 5% buffer for view
                with columns[0]:
                    min_y = st.number_input(
                        label="y-axis min", key=f"{hash_id}_{_attribute}_min", value=default_min_y if default_min_y != 0.0 else -0.05
                    )
                with columns[1]:
                    max_y = st.number_input(
                        label="y-axis max", key=f"{hash_id}_{_attribute}_max", value=default_max_y if default_max_y != 0.0 else 1.05
                    )
                _output_dict[_attribute] = {"range": [min_y, max_y]}
            if st.form_submit_button("Submit"):
                sess_state.custom_y_axes_dict[agent_id] = _output_dict.copy()

    return sess_state


def _plot_layout_aggregate_option(agent_steps_table: pd.DataFrame, plot_kwargs: dict) -> tuple[pd.DataFrame, dict]:
    """Appends the 95% CI columns to `agent_steps_table` and update `plot_kwargs` accordingly.

    Parameters
    ----------
    agent_steps_table : pd.DataFrame
        The agent steps table from EpisodeArtifactTable or a subset of it.
    plot_kwargs : dict
        The plotly kwargs thus far.

    Returns
    -------
    pd.DataFrame
        The modified `agent_steps_table`
    Dict
        The updated plot_kwargs
    """

    agent_steps_table = append_confidence_interval_columns(
        dataframe=agent_steps_table, steps_col=plot_kwargs["x"], attribute_name_col="attribute_name", attribute_value_col="attribute_value"
    )
    plot_kwargs.update({"facet_col": "attribute_name", "facet_col_wrap": 1, "y": "attribute_value_mean"})

    return agent_steps_table, plot_kwargs


def _plot_layout_different_plots_option(agent_steps_table: pd.DataFrame, plot_kwargs: dict) -> dict:
    """Updates the plot_kwargs

    Parameters
    ----------
    agent_steps_table : pd.DataFrame
        The agent steps table from EpisodeArtifactTable or a subset of it.
    plot_kwargs : dict
        The plotly kwargs thus far.

    Returns
    -------
    Dict
        The updated plot_kwargs
    """
    n_selected_attributes = agent_steps_table["attribute_name"].nunique()
    plot_kwargs.update(
        {
            "facet_col": "attribute_name",
            "facet_col_wrap": 1,
            "y": "attribute_value",
            "height": 300 * n_selected_attributes,
            "facet_row_spacing": 0.03,
        }
    )
    return plot_kwargs


def _plot_layout_same_plot_option(plot_kwargs: dict) -> dict:
    """Updates the plot_kwargs

    Parameters
    ----------
    plot_kwargs : dict
        The plotly kwargs thus far.

    Returns
    -------
    Dict
        The updated plot_kwargs
    """
    plot_kwargs.update({"color": "attribute_name", "y": "attribute_value", "height": 500})
    return plot_kwargs


def _plot_layout_show_all_option(agent_steps_table: pd.DataFrame, plot_kwargs: dict) -> dict:
    """Updates the plot_kwargs

    Parameters
    ----------
    agent_steps_table : pd.DataFrame
        The agent steps table from EpisodeArtifactTable or a subset of it.
    plot_kwargs : dict
        The plotly kwargs thus far.

    Returns
    -------
    Dict
        The updated plot_kwargs
    """
    n_selected_attributes = agent_steps_table["attribute_name"].nunique()
    plot_kwargs.update(
        {
            "color": "test_case",
            "y": "attribute_value",
            "facet_col": "attribute_name",
            "facet_col_wrap": 1,
            "height": 500 * n_selected_attributes,
        }
    )
    return plot_kwargs


def _section_headers():
    st.markdown("## Agent Steps")
    st.markdown(
        """
    - **Array elements**: Any observations or actions that were originally array elements are flattened in the table.
      For example, if an attribute named `obs_sensor` is originally: [0, 1, 0], it will be flattened to
      `obs_sensor_0 = 0`, `obs_sensor_1=1`,`obs_sensor_2=0` in the flattened table.
    - **Repeated Observations**: Repeated observations are stored in CoRL as a `List[Dict[str, Any]]`. Again, this is flattened
      in the table. For example, given a repeated observation sensor in the format:
      `ObserveSensorRepeated_SensorExample = [{"position": [1,2], "angle":90}, {"position": [10,20], "angle":45}]`,
      the corresponding on rows in the table are:
        ```
        # These all correspond to the first element of the top-level list
        ObserveSensorRepeated_SensorExample_position[0]_0 = 1 # First element of position
        ObserveSensorRepeated_SensorExample_position[0]_1 = 2 # Second element of position
        ObserveSensorRepeated_SensorExample_angle[0] = 90

        # Second element of the top-level list
        ObserveSensorRepeated_SensorExample_position[1]_0 = 10 # First element of position
        ObserveSensorRepeated_SensorExample_position[1]_1 = 20 # Second element of position
        ObserveSensorRepeated_SensorExample_angle[0] = 45
        ```
    - **Multiplatform Agents**: The sensors and controls for multiplatform agents will have the platform name in the part name.
    """
    )
