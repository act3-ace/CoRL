"""
---------------------------------------------------------------------------

This is a US Government Work not subject to copyright protection in the US.
The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from ray.rllib.env.env_context import EnvContext
from ray.rllib.policy.sample_batch import SampleBatch
from stqdm import stqdm

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.runners.section_factories.engine.rllib.episode_artifact_logging_callback import EVAL_CHECKPOINTS_SUBDIR
from corl.evaluation.runners.section_factories.engine.rllib.rllib_trainer import ray_context
from corl.visualization.network_explainability.network_explainability import AgentCheckpoint
from corl.visualization.network_explainability.network_explainability_plotter import PPOTorchNetworkExplainabilityPlotter

use_container_width = True


def write_code_block(text: str):
    """Helper function to wrap text in code-style block"""
    st.markdown(
        f"""
    ```
    {text}
    ```
    """
    )


def convert_counts_to_percentage(input_table: pd.DataFrame, count_column: str) -> pd.Series:
    """Helper function to convert counts to percentages"""
    return 100.0 * input_table[count_column] / input_table[count_column].sum()


def st_dataframe_pagination(table_key: str, pages_dict: dict, tables_dict: dict | None = None, dataframe: pd.DataFrame | None = None):
    """Streamlit dataframe widget for pandas dataframe with pagination
    This relies on the session state variable `pages_dict` to be
    defined.

    Parameters
    ----------
    table_key : str
        The key corresponding to the tables dict and / or pages_dict
    pages_dict : dict
        A dictionary that stores which page the table view is on
    tables_dict : Optional[dict], optional
        A dictionary of all the tables, optional, by default None
    dataframe : Optional[pd.DataFrame], optional
        The dataframe to show, by default None
    """
    VIEW_LIMIT = 1000
    if tables_dict:
        _dataframe: pd.DataFrame = tables_dict[table_key]

    elif dataframe is not None:
        _dataframe = dataframe
    else:
        raise ValueError("A dataframe or tables_dict needs to be provided.")

    _dataframe = _dataframe.copy()
    _dataframe = _dataframe.reset_index(drop=True)
    total_rows = len(_dataframe)
    if len(_dataframe) < VIEW_LIMIT:
        st.dataframe(_dataframe, use_container_width=use_container_width)

    else:
        hash_id = pd.util.hash_pandas_object(_dataframe).sum()
        max_pages = np.ceil(len(_dataframe) / VIEW_LIMIT)
        current_page = pages_dict.get(table_key, 0)
        offset = np.clip(VIEW_LIMIT * current_page, a_min=0, a_max=total_rows - VIEW_LIMIT)
        subset = _dataframe.loc[offset : offset + VIEW_LIMIT]
        st.dataframe(subset, use_container_width=use_container_width)

        columns = st.columns(6)
        with columns[-3]:
            if st.button(label="Prev Page", key=f"{table_key}_prev_page_{current_page}_{hash_id}"):
                pages_dict[table_key] = max(current_page - 1, 0)
                st.experimental_rerun()
        with columns[-2]:
            if st.button(label="Next Page", key=f"{table_key}_next_page_{current_page}_{hash_id}"):
                pages_dict[table_key] = min(current_page + 1, max_pages - 1)
                st.experimental_rerun()
        with columns[-1]:
            st.markdown(f"Showing: {int(offset)}:{int(offset+VIEW_LIMIT)} of {total_rows}")
    return pages_dict


def confidence_interval(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)


def init_network_explainability_module(
    agent_checkpoints: list[AgentCheckpoint],
    episode_artifact_files: list[Path],
    episode_artifact_list: list[EpisodeArtifact] | None = None,
    env_config_path: str | Path | None = None,
    env_config: dict | EnvContext | None = None,
    verbose=True,
    policy_network_input_map: dict | None = None,
) -> PPOTorchNetworkExplainabilityPlotter:
    """Initializes and returns the network explianability module

    Parameters
    ----------
    agent_checkpoints : List[AgentCheckpoint]
        A list of agent checkpoints
    episode_artifact_files : Optional[List[Union[str, Path]]], optional
        A list of episode artifact files, by default None
    episdoe_artifact_list : Optional[List[EpisodeArtifact]], optional
        A list of episode artifacts already loaded, by default None
    env_config_path : Optional[Union[str, Path]], optional
        The path to the environment configuration, by default None
    env_config : Optional[Union[Dict, EnvContext]], optional
        The env context, already loaded, by default None
    verbose : bool, optional
        When True, verbose output, by default True

    Returns
    -------
    PPOTorchNetworkExplainabilityPlotter
        The network visaulizer class.
    """
    if env_config is None:
        with open(env_config_path, "rb") as file_obj:  # type: ignore
            env_config = pickle.load(file_obj)  # noqa: S301
    with ray_context(local_mode=True, include_dashboard=False, num_cpus=1):
        explainer_viz = PPOTorchNetworkExplainabilityPlotter(agent_checkpoints=agent_checkpoints, env_config=env_config)
        if verbose:
            st.text("Generating Policy Network Visualizations...")

        if episode_artifact_list:
            for episode_artifact in stqdm(episode_artifact_list):
                explainer_viz.append_feat_importance_gradient_based(episode_artifact, policy_network_input_map)
                explainer_viz.append_vf_residuals(episode_artifact)

        elif episode_artifact_files:
            for file in stqdm(episode_artifact_files):
                with open(file, "rb") as file_obj:
                    episode_artifact = pickle.load(file_obj)  # noqa: S301
                explainer_viz.append_feat_importance_gradient_based(episode_artifact, policy_network_input_map)
                explainer_viz.append_vf_residuals(episode_artifact)
        else:
            raise ValueError("Must provide either `episode_artifact_files` or `episode_artifact_list")

    return explainer_viz


def st_directory_picker(title: str) -> tuple[Path, str | None]:
    """Generates a text input for selecting a directory

    Parameters
    ----------
    title : str
    initial_path : Path, optional
        The initial path, by default Path()

    Returns
    -------
    Path
        The selected directory.
    """
    st.markdown("#### " + title)

    columns = st.columns(2)

    with columns[0]:
        selected_path = Path(st.text_input("Select a directory:"))

    sub_dirs = [f.name for f in selected_path.iterdir() if f.is_dir() and (not f.stem.startswith(".") and not f.stem.startswith("__"))]
    chosen_subdir = None
    with columns[1]:
        if sub_dirs:
            chosen_subdir = st.selectbox("Choose Run", sorted(sub_dirs))
        else:
            st.markdown("#")
            st.markdown(":red[No subdirectories found]")

    return selected_path, chosen_subdir


def read_eval_experiment_file(experiment_dir):
    """Returns a dictionary corresponding to the experiment set up and
    the name of the yaml file read"""
    yaml_file = list(Path(experiment_dir).glob("*.y*ml"))
    assert len(yaml_file) == 1
    with open(yaml_file[0], "rb") as stream:
        try:
            experiment_config = yaml.safe_load(stream)
            return experiment_config, yaml_file[0]
        except yaml.YAMLError as exc:
            print(exc)
            return None, None


def create_selection_grid_single_and_self_policy(num_checkpoints, training_run_dir, main_policy):
    """Helper function to create the selection grid for single policy and self-play.

    Parameters
    ----------
    num_checkpoints : int
        Number of checkpoints to inspect
    training_run_dir : Union[str, Path]
        The training run directory.
    main_policy : str
        The name of the main policy

    Returns
    -------
    List[Dict[str, Any]]
        A list of the selections where each selection is a dictionary containing
        an AgentCheckpoint and epoch selected.
    """
    training_run_dir = Path(training_run_dir)
    # The grid will be x rows by 2 columns
    assert training_run_dir.exists(), f"Directory does not exist: {training_run_dir}"
    num_columns = min(num_checkpoints, 2)
    num_rows = int(np.ceil(num_checkpoints / num_columns))

    show_columns = num_columns
    items_displayed = 0
    return_selections = []
    _unqiue_selections = []
    for row in range(num_rows):
        _columns = st.columns(2)
        # Shows at most 2 columns
        for col in range(show_columns):
            with _columns[col]:
                checkpoint_options = _checkpoint_options(training_run_dir)
                checkpoint_selected = st.selectbox("Pick a Checkpoint", checkpoint_options, key=f"checkpoint_{col}_{row}")
            # Keeps track of how many items have already been displayed
            items_displayed += 1
            show_columns = min(num_checkpoints - items_displayed, num_columns)
            if checkpoint_selected:
                checkpoint_dir = training_run_dir.joinpath(EVAL_CHECKPOINTS_SUBDIR, checkpoint_selected, "policies", main_policy)
                if checkpoint_selected not in _unqiue_selections:
                    _unqiue_selections.append(str(checkpoint_selected))
                    return_selections.append(
                        {
                            "agent_checkpoint": AgentCheckpoint(agent_name=main_policy, checkpoint_dir=checkpoint_dir),
                            "epoch_selected": checkpoint_selected,
                        }
                    )
                else:
                    st.text(f"Ignored Selections due to Duplication: {checkpoint_selected}")
    return return_selections


def create_selection_grid_policy_is_agent(num_checkpoints, training_run_dir):
    """Creates the selection grid for PolicyIsAgent. Allows the user to specify
    a policy for each checkpoint.

    Parameters
    ----------
    num_checkpoints : int
        Number of checkpoints to inspect
    training_run_dir : Union[str, Path]
        The training run directory.
    Returns
    -------
    List[Dict[str, Any]]
        A list of the selections where each selection is a dictionary containing
        an AgentCheckpoint and epoch selected.
    """
    training_run_dir = Path(training_run_dir)
    _unique_selections = []
    # The grid will be x rows by 2 columns
    assert training_run_dir.exists(), f"Directory does not exist: {training_run_dir}"
    num_rows = num_checkpoints
    return_selections = []
    for row in range(num_rows):
        _columns = st.columns(2)
        with _columns[0]:
            checkpoint_options = _checkpoint_options(training_run_dir)
            checkpoint_selected = st.selectbox("Pick a Checkpoint", checkpoint_options, key=f"checkpoint_{row}")
        if checkpoint_selected:
            with _columns[1]:
                policy_options = [
                    path.name for path in training_run_dir.joinpath(EVAL_CHECKPOINTS_SUBDIR, checkpoint_selected, "policies").glob("*")
                ]
                policy_options.sort()
                policy_selected = st.selectbox("Pick a policy", policy_options, key=f"policy_{row}")
                checkpoint_dir = None
                if policy_selected:
                    checkpoint_dir = training_run_dir.joinpath(EVAL_CHECKPOINTS_SUBDIR, checkpoint_selected, "policies", policy_selected)
            _selection_key = f"{policy_selected}-{checkpoint_selected}"
            if _selection_key not in _unique_selections and checkpoint_dir:
                return_selections.append(
                    {
                        "agent_checkpoint": AgentCheckpoint(agent_name=policy_selected, checkpoint_dir=checkpoint_dir),
                        "epoch_selected": checkpoint_selected,
                    }
                )
                _unique_selections.append(_selection_key)
            else:
                st.text(f"Ignored Selection due to Duplication, checkpoint: {checkpoint_selected}, policy: {policy_selected}")
    return return_selections


@st.cache_resource
def generate_grad_based_feat_importance(input_fig):
    """Helper function to generate gradient based feature importance"""
    st.markdown("### Feature Importance: Gradient Based")
    st.plotly_chart(input_fig, theme=None, use_container_width=use_container_width)
    st.markdown(
        "**Figure**: Average Normalized Feature Importance: aggregated across all test cases. \
        The gradient-based feature importance depicts the gradient of the network output w.r.t. each network \
            input. The absolute value of the raw gradient-based feature importance is taken before it is averaged \
            and normalized. Observations represented by a Discrete / MultiDiscrete space are transformed into one-hot encoded vectors \
            before they are ingested by RLLib. These vectors are consolidated (summed) in this visualization.  \
            When the policy-network input contains more than the current observation (e.g. a series of observations), \
            the absolute value of the gradients are additionally summed across the time-axis."
    )
    return input_fig


@st.cache_resource
def generate_vf_residuals_dist(input_fig):
    """Helper function to generate gradient based feature importance"""
    st.markdown("### Value Function Residuals")
    st.plotly_chart(input_fig, theme=None, use_container_width=use_container_width)
    st.markdown("**Figure**: The residual of the value function.")
    return input_fig


def _checkpoint_options(training_run_dir):
    """Helper function to show the checkpoint options."""
    checkpoint_options = [path.name for path in training_run_dir.glob(f"{EVAL_CHECKPOINTS_SUBDIR}/epoch_*")]
    checkpoint_options.sort()
    return checkpoint_options


def _check_required_subdirectories_policy_viz(root_dir):
    """Helper function that checks the required subdirectories for the policy
    visualization. Produces a helpful error when a directory is not found."""
    root_dir = Path(root_dir)
    error_msg = f"Run: {root_dir} is missing {EVAL_CHECKPOINTS_SUBDIR}"
    assert root_dir.joinpath(EVAL_CHECKPOINTS_SUBDIR).exists(), error_msg
    error_msg = f"Run: {root_dir} is missing trajectories"
    assert root_dir.joinpath("trajectories").exists(), error_msg


def _group_checkpoints_by_input_output(input_output_dict):
    """Given an input_output_dict with keys checkpoint, input and output,
    groups by input and output and returns grouped checkpoints"""
    dataframe = pd.DataFrame(input_output_dict)
    dataframe["checkpoint"] = dataframe["checkpoint"].apply(lambda row: [row])
    dataframe = dataframe.groupby(["input", "output"]).agg({"checkpoint": sum})
    return dataframe["checkpoint"].values.tolist()


def checkbox_grid(
    possible_choices: list, key_prefix: str, prev_selected: list | None = None, possible_choices_return_value: list | None = None
):
    """
    Creates a block of checkboxes for user selection
    Parameters
    ----------
    possible_choice_list : List
        A list of possible choices
    key_prefix : str
        The prefix for the checkbox key. When this function is called more than once
        a unique key_prefix must be provided each time to satisfy strealmit.
    prev_selected : List, optional
        All values provided in this list will be set to True. Defaults to None.
    Returns
    -------
    List
        A list of selected choices
    """
    if possible_choices_return_value:
        assert len(possible_choices) == len(
            possible_choices_return_value
        ), "The length of possible choices and possible_choices_return_value must be the same."

    if prev_selected is None:
        prev_selected = []

    selected_choices = []
    num_columns = 3
    columns = st.columns(num_columns)
    items_per_column = int(np.ceil(len(possible_choices) / num_columns))
    # Create check boxes. The checkboxes are displayed in `num_columns` columns
    column_counter = 0
    for idx, attribute in enumerate(possible_choices):
        with columns[column_counter]:
            state = st.checkbox(label=attribute, value=attribute in prev_selected, key=f"{key_prefix}_{attribute}")
        if state:
            if possible_choices_return_value:
                selected_choices.append(possible_choices_return_value[idx])
            else:
                selected_choices.append(attribute)
        if (idx + 1) % items_per_column == 0:
            column_counter += 1
    return selected_choices


def center_plotly_figure(figure):
    """Helper function to center the plotly figure."""
    columns = st.columns(3)
    for idx, column in enumerate(columns):
        with column:
            if idx == 1:
                st.plotly_chart(figure, use_container_width=False, theme=None)
            else:
                pass


def text_section_policy_network_input(agent_checkpoints: list[AgentCheckpoint]) -> dict[str, str]:
    """Generates an input section for the policy network input names as free-form text.
    The default value is `SampleBatch.OBS`"""
    st.markdown(
        f"""**Specify the poilcy network inputs for each trained policy.**
    This is typically {SampleBatch.OBS}, for the default model used in `PPOTrainer`."""
    )
    policy_input_map: dict = {}
    n_columns = min(len(agent_checkpoints), 5)
    column_idx = 0
    columns = st.columns(n_columns)

    for agent_checkpoint in agent_checkpoints:
        agent_id = agent_checkpoint.agent_name
        with columns[column_idx]:
            policy_input_map[agent_id] = st.text_input(label=agent_id, key=agent_id, value=SampleBatch.OBS)
        column_idx += 1
        # Once n_columns columns are reached, create a new row
        if column_idx % n_columns == 0:
            column_idx = 0
            columns = st.columns(n_columns)

    st.write(policy_input_map)

    return policy_input_map


def ui_dataframe_filter(dataframe: pd.DataFrame, markdown_title: str | None = None) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    """
    if markdown_title:
        st.markdown(markdown_title)
    dataframe = dataframe.copy()
    with st.container():
        to_filter_columns = st.multiselect("Filter dataframe on", dataframe.columns)
        for column in to_filter_columns:
            _, right_column = st.columns((1, 20))

            # For numeric values, show a slider
            if is_numeric_dtype(dataframe[column]):
                _min = float(dataframe[column].min())
                _max = float(dataframe[column].max())
                step = (_max - _min) / 100
                user_num_input = right_column.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                dataframe = dataframe[dataframe[column].between(*user_num_input)]
            # Treat columns with < 10 unique values as categorical
            elif is_categorical_dtype(dataframe[column]) or dataframe[column].nunique() < 10:
                user_cat_input = right_column.multiselect(
                    f"Values for {column}",
                    dataframe[column].unique(),
                    default=list(dataframe[column].unique()),
                )
                dataframe = dataframe[dataframe[column].isin(user_cat_input)]
            # Otherwise, cast as string and allow users to allow user to use substring or regex
            else:
                user_text_input = right_column.text_input(f"Substring or regex in {column}")
                if user_text_input:
                    dataframe = dataframe[dataframe[column].astype(str).str.contains(user_text_input)]
        st.dataframe(dataframe, use_container_width=True)
    return dataframe
