"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Network Explainability Plotter
"""

import typing

import numpy as np
import pandas as pd
import plotly.express as px
from ray.rllib.utils.typing import AgentID

from corl.visualization.network_explainability.network_explainability import PPOTorchNetworkExplainability


class PPOTorchNetworkExplainabilityPlotter(PPOTorchNetworkExplainability):
    """
    Renders plots for various explainability tools.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.color_palette = px.colors.qualitative.Alphabet
        self.height = 600

    def render_agg_feat_importance_grad_based(self, feats_dict: dict[str, typing.Any] | None = None, **kwargs):
        """Plotting wrapper for `get_agg_feat_importance_grad_based`, see method for
        input parameters.

        Parameters
        ----------
        feats_dict : Optional[Dict[str, Any]]
            An optional features dict to pass in. When one is not provided, calls
            `self.get_agg_feat_importance_grad_based(**kwargs), by default None

        Returns
        -------
        plotly.graph_objs.Figure
            Figure object.
        """
        if feats_dict is None:
            _feats_dict: dict[AgentID, dict[str, typing.Any]] = self.get_agg_feat_importance_grad_based(**kwargs)
        else:
            _feats_dict = feats_dict

        dataframe_list = []
        for policy_id, feats in _feats_dict.items():
            # Manipuate the data into a long formatted dataframe for plotting
            dataframe = pd.DataFrame(feats)
            dataframe = dataframe.reset_index()
            dataframe.rename(columns={"index": "input_obs"}, inplace=True)
            dataframe = pd.melt(dataframe, value_name="feat_importance", id_vars=["input_obs"], var_name="policy_output")
            dataframe["policy_id"] = policy_id
            dataframe_list.append(dataframe)
        dataframe_all = pd.concat(dataframe_list, axis=0, ignore_index=True)

        fig = px.bar(
            data_frame=dataframe_all,
            x="feat_importance",
            y="policy_output",
            color="input_obs",
            facet_col="policy_id",
            orientation="h",
            height=self.height,
            color_discrete_sequence=self.color_palette,
            labels={
                "input_obs": "Policy Input",
                "policy_output": "Network Output",
                "feat_importance": "Average Normalized Feature Importance",
            },
            facet_col_wrap=2,
        )

        fig.update_layout(yaxis={"automargin": True}, font={"size": 16})
        return fig

    def render_value_function_residual_dist(self, vf_residuals: dict[str, typing.Any] | None = None):
        """Plotter to render the distribution of the value function residuals.

        Parameters
        ----------
        vf_residuals : typing.Optional[typing.Dict[str, typing.Any]], optional
            The value function residuals where the key is a reference name (e.g. policy, agent or some other descriptor),
            when vf_residuals is not provided, will call self.vf_residuals to retrieve, by default None

        Returns
        -------
        plotly.graph_objs.Figure
            Figure object.
        """
        dfs = []
        if vf_residuals is None:
            vf_residuals = self.vf_residuals
        for name, residuals_list in vf_residuals.items():
            one_agent_df = pd.DataFrame(np.concatenate(residuals_list), columns=["vf_residuals"])
            one_agent_df["policy_id"] = name
            dfs.append(one_agent_df)
        dataframe = pd.concat(dfs, axis=0, ignore_index=True)
        return px.histogram(
            dataframe,
            x="vf_residuals",
            color="policy_id",
            color_discrete_sequence=self.color_palette,
            marginal="box",
            height=self.height,
            labels={"policy_id": "Policy", "vf_residuals": "Value Targets - Value Predictions"},
        )
