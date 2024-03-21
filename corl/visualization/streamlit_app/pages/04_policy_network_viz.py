"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Policy Network Visualization
"""
import streamlit as st

from corl.evaluation.recording.folder import EpisodeArtifactLoggingCallbackLoader, FolderRecord
from corl.visualization.streamlit_app.default_layout import DefaultStreamlitPage
from corl.visualization.streamlit_app.eval_while_train_layout import StreamlitPageEvaluationWhileTraining

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    # #########################################################################
    # Policy Network Visualization Tool.
    # #########################################################################
    st.markdown("# Policy Network Visualization")
    st.warning(
        """This page may cause the app to crash.
        Under the hood, this initializes the simulation environment, loads in your trained policies
        and initializes Ray. When the simulation environment is large, it seems to cause the app
        to either freeze or crash.
        """
    )
    with st.form("eval-selection"):
        corl_eval = "CoRL Custom Evaluation"
        eval_while_train = "Evaluation Generated while Training"

        selected_output = st.radio(label="Select Evaluation Type", options=(corl_eval, eval_while_train), horizontal=True)
        submitted = st.form_submit_button("Done")

    if submitted:
        if selected_output == corl_eval:
            st.session_state.policy_network_app = DefaultStreamlitPage(file_loader=FolderRecord)
        else:
            st.session_state.policy_network_app = StreamlitPageEvaluationWhileTraining(file_loader=EpisodeArtifactLoggingCallbackLoader)

    if hasattr(st.session_state, "policy_network_app"):
        st.session_state.policy_network_app.display_directory_picker()
        if st.button("Select Run") or hasattr(st.session_state, "selected_run"):
            st.session_state.selected_run = True
            st.session_state.policy_network_app.display_policy_network_viz_section()
