"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Entrypoint for Streamlit App
"""
import jsonargparse
import streamlit as st

from corl.parsers.yaml_loader import load_file
from corl.visualization.streamlit_app.default_layout import DefaultStreamlitPage, Functor


@st.cache_data
def read_config_file(yaml_file: str) -> dict:
    """Returns the parsed configuration file as a dictionary

    Parameters
    ----------
    yaml_file : str
        The path to the yaml file.

    Returns
    -------
    dict:
        The parsed configurations.
    """
    return load_file(yaml_file)


def _parse_display_trajectory_config(config: dict) -> dict:
    """Parses the configs for the trajectory animation section of streamlit"""
    name = DefaultStreamlitPage.display_trajectory_animation_section.__name__
    if name not in config["streamlit_layout"]:
        return config

    config["streamlit_layout"][name]["animator"] = Functor(**config["streamlit_layout"][name]["animator"])
    config["streamlit_layout"][name]["frame_axis_bounds"] = Functor(
        **config["streamlit_layout"][name]["frame_axis_bounds"]
    ).create_functor_object()
    return config


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the yaml file containing the plotly configurations", type=str, default=None)
    args = parser.parse_args()

    st.set_page_config(layout="wide", page_title="Introduction")
    st.session_state.layout_config = {}
    if args.config:
        st.session_state.layout_config = _parse_display_trajectory_config(read_config_file(yaml_file=args.config))

    st.markdown("# Overview")
    st.markdown(
        """
        This application was developed to inspect evaluation trajectories stored in the episode artifacts.
        - The Evaluation From Training page allows the user to inspect episode artifacts generated from a training run
        This implies a logging callback was included as part of training and evaluation was enabled through the `rllib_config.yml`.
        - The Corl Evaluation page allows user to inspect episode artifacts generated from a custom corl evaluation.
        - This streamlit app takes as an optional input, a configuration file that defines the trajectory animation section. See
        `config/streamlit_app/default_streamlit_config.yml` for an example with Pong.
        """
    )
    # NOTE: in order for the argparser to parse correctly and set the `layout_config`
    # this page must fully load before the subsequent pages are loaded.
    st.session_state.main_page_loaded = True
