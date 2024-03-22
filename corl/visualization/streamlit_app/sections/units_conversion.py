"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Functions for the unit options in the streamlit app
"""

import pandas as pd
import streamlit as st

from corl.libraries.units import corl_get_ureg
from corl.visualization.streamlit_app.utils import write_code_block


def unit_selection_and_table_update(tables_dict: dict[str, pd.DataFrame], obs_units_flattened: dict) -> dict:
    """Streamlit section that displays the unit selector and returns the
    tables dict with the units converted for the AgentSteps and SpaceData tables.

    Parameters
    ----------
    tables_dict : Dict
        This is expected to be EpisodeArtifactTables.tables
    obs_units_flattened : Dict[str, str]
        A dictionary that maps the attribute name to the unit name (str)

    Returns
    -------
    Dict[str, pd.DataFrame]
        The updated tables_dict
    """
    st.markdown("## Unit Conversion")
    # Check if there are any values with real units
    none_units = [unit_str == "dimensionless" for _, unit_str in obs_units_flattened.items()]
    if len(obs_units_flattened) == 0 or all(none_units):
        st.warning("No observation unit information was found or all units are dimensionless types.")
        return tables_dict

    st.markdown("**Note**: This will change both the SpaceData table and the AgentSteps table.")
    ignore_dimensions = ["dimensionless"]
    present_dimensions = []
    # attribute_name -> dimension
    attribute_name_dimension_map = {}
    obs_units_flattened = obs_units_flattened.copy()
    # Gets a list of relevant dimensions
    for attribute_name, unit_str in obs_units_flattened.items():
        if unit_str not in ignore_dimensions:
            attribute_name_dimension_map[attribute_name] = unit_str
            if unit_str not in present_dimensions:
                present_dimensions.append(unit_str)

    # Creates a dropbox selector for each (unit) dimension
    user_selected_units = {}
    columns = st.columns(4)
    col_idx = 0
    for dimension_name in present_dimensions:
        if dimension_name in ignore_dimensions:
            continue

        ureg = corl_get_ureg()
        options = []
        for u in ureg.units:
            if corl_get_ureg().get_unit(dimension_name).is_compatible_with(u) and corl_get_ureg().get_unit(u) not in options:
                options.append(str(corl_get_ureg().get_unit(u)))

        # dimension_name -> desired_units
        if col_idx % 4 == 0:
            col_idx = 0
            columns = st.columns(4)
        with columns[col_idx]:
            user_selected_units[dimension_name] = st.selectbox(label=dimension_name, options=options, key=f"selection_{dimension_name}")
        col_idx += 1

    if st.button("Convert units"):
        write_code_block("Converting units in agent steps...")
        # attribute_name -> new units
        new_units_map = {
            attribute_name: user_selected_units[dimension_name] for attribute_name, dimension_name in attribute_name_dimension_map.items()
        }

        old_to_new_units: dict[str, str] = {obs_units_flattened[k]: v for k, v in new_units_map.items() if v is not None}
        update_units_flattened_agent_steps(tables_dict=tables_dict, old_to_new_units_dict=old_to_new_units)

        write_code_block("Updating units in space table...")
        tables_dict = update_units_space_table(tables_dict=tables_dict, new_units_map=new_units_map)

    return tables_dict


def update_units_flattened_agent_steps(tables_dict: dict, old_to_new_units_dict: dict[str, str]) -> dict:
    """Converts the united values in the agents steps table.

    Parameters
    ----------
    old_to_new_units_dict: dict
        A mapping from old units to new units
    """
    dataframe = tables_dict["FlattenedAgentStep"].copy()
    if not dataframe.empty:

        def apply_unit_conversion_to_row(row):
            tmp = corl_get_ureg().Quantity(row["attribute_value"], units=row["units"])
            return tmp.m_as(old_to_new_units_dict[row["units"]])

        idx_has_units = dataframe["units"].notnull()
        dataframe.loc[idx_has_units, "attribute_value"] = dataframe.loc[idx_has_units].apply(
            lambda row: apply_unit_conversion_to_row(row) if row["units"] in old_to_new_units_dict else row["attribute_value"], axis=1
        )
        dataframe["units"] = dataframe["units"].apply(lambda x: old_to_new_units_dict.get(x, x))
        tables_dict["FlattenedAgentStep"] = dataframe
    return tables_dict


def update_units_space_table(tables_dict: dict, new_units_map: dict) -> dict:
    """Updates the units in the Space Data table

    Parameters
    ----------
    new_units_map : dict
        The key must be an attribute name and the value must be the unit to convert the value to.
    """
    dataframe = tables_dict["SpaceData"].copy()
    if not dataframe.empty:

        def apply_unit_conversion_to_row(row, prefix, column):
            tmp = corl_get_ureg().Quantity(row[f"{prefix}{column}"], units=row["_prev_units"])
            return tmp.m_as(row["units"])

        dataframe["_prev_units"] = dataframe["units"].copy()

        dataframe["units"] = dataframe.apply(lambda row: new_units_map.get(row["attribute_name"], row["units"]), axis=1)
        for column in ["min", "max"]:
            # Convert the "normalized columns" that have not been normalized when units exist
            dataframe[f"normalized_{column}"] = dataframe.apply(
                lambda row, column=column: apply_unit_conversion_to_row(row, "normalized_", column)
                # Only performs conversions when the normalized
                # and unnormalized columns match and units exist
                if row["min"] == row["normalized_min"] and row["max"] == row["normalized_max"] and row["_prev_units"]
                else row[f"normalized_{column}"],
                axis=1,
            )
            # Converts the unnormalized columns. NOTE: this must go after the normalized column.
            dataframe[column] = dataframe.apply(
                lambda row, column=column: apply_unit_conversion_to_row(row, "", column) if row["_prev_units"] else row[column], axis=1
            )

        dataframe = dataframe.drop(columns=["_prev_units"])
        tables_dict["SpaceData"] = dataframe

    return tables_dict
