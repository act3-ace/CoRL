"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Trajectory Animation Class
"""
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm
from plotly.subplots import make_subplots

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.visualization.base_animator import BaseAnimator, FrameAxisBounds
from corl.visualization.platform_plotting_deserializer import PlatformDeserializerPlotlyAnimation


class PlotlyTrajectoryAnimation(BaseAnimator):
    """Produces a trajectory animation given a set of PlatformSteps.
    This can handle both 2D and 3D shapes. The colors of each platform is inferred
    by the name of each platform. For example, when a platform is named "blue0",
    it will be blue. When more than one platform is named with the same color,
    they will all be plotted in the same color. When no color is provided as part
    of the platform name, a random color is chosen.

    """

    def __init__(
        self,
        episode_artifact: EpisodeArtifact,
        platform_serializer_map: dict[str, str],
        progress_bar_func: Callable = tqdm.tqdm,
        platform_color_map: dict[str, str] | None = None,
    ):
        super().__init__(
            episode_artifact=episode_artifact, platform_serializer_map=platform_serializer_map, platform_color_map=platform_color_map
        )

        self.progress_bar_func = progress_bar_func

        self._transition_duration_ms = 10
        self._frame_duration_ms = 30

        self.figure_width = 900
        self.figure_height = 900
        self.frame_axis_bounds: FrameAxisBounds | None = None

    def set_frame_axis_bounds(self, frame_axis_bounds: FrameAxisBounds):
        """Sets the frame axis bounds.

        Parameters
        ----------
        frame_axis_bounds : FrameAxisBounds
            The coordinates of the frame axis
        """
        assert isinstance(frame_axis_bounds, FrameAxisBounds)
        if self.get_dimensionality() == 2:
            assert (
                frame_axis_bounds.zbounds is None
            ), """
            Dimension mismatch between the world and the frame axis bounds.
            z-coordinates should be None for a 2-dimensional world."""

        self.frame_axis_bounds = frame_axis_bounds

    @property
    def platform_deserializer(self):
        """Platform deserializer as prop"""
        deserializer = self.get_deserializer()
        assert issubclass(
            deserializer, PlatformDeserializerPlotlyAnimation
        ), f"Deserializer is expected to subclass: {PlatformDeserializerPlotlyAnimation.__name__}, deserializer: {deserializer} is invalid"
        return deserializer()

    def get_metadata_names(self):
        """Returns a list of the metadata that is available to display as text
        above the platform"""
        metadata_keys = []
        # This is a list of serialized platform data for each platform for one step
        for platform_serialized_data in self.config.episode_artifact.steps[0].platforms:
            metadata = self.platform_deserializer.get_metadata_for_text_display(platform_serialized_data)
            metadata_keys.extend(list(metadata.keys()))
        metadata_keys = list(set(metadata_keys))
        metadata_keys.sort()
        return metadata_keys

    def get_dimensionality(self):
        """Returns the expected dimensionality of the world."""
        return self.platform_deserializer.get_dimensionality(serialized_data=self.config.episode_artifact.steps[0].platforms[0])

    @staticmethod
    def _check_and_parse_position(position: list | tuple, position_units: list | tuple):
        assert len(position) in [
            1,
            2,
            3,
        ], f"""
        The position must either be 1- 2- or 3- dimensional.
        Found: {len(position)} dimensions in {position}"""
        coordinate_keys = ["x", "y", "z"]
        parsed_position = {}
        for idx, val in enumerate(position):
            key_name = f"{coordinate_keys[idx]}[{position_units[idx]}]"
            parsed_position[key_name] = val
        return parsed_position

    def generate_animation(  # noqa: PLR0915
        self,
        selected_metadata: list[str] | None = None,
    ) -> tuple[go.Figure, list[go.Figure], dict[str, pd.DataFrame]]:
        """Generates the animation

        Parameters
        ----------
        selected_metadata : List[str]
            A list of selected metadata to display above each platform

        Returns
        -------
        go.Figure
            The figure as an animation
        List[go.Figure]
            A list of the frames that make up the animation
        Dict[str, pd.DataFrame]
            A table of the underlying data for each platform.
        """

        # #####################################################################
        # Configure Plotly Figures
        # #####################################################################
        expected_dims = self.get_dimensionality()

        # Redraw is False for 2D - this speeds up the animation
        redraw = expected_dims != 2 or 1
        trajectory_position = {"row": 1, "col": 1}
        subplot_kwargs = {"rows": 1, "cols": 1, "specs": [[{"type": "scatter" if expected_dims <= 2 else "scene"}]]}
        fig = make_subplots(**subplot_kwargs)
        fig.update_layout({"width": self.figure_width, "height": self.figure_height})
        # When region bounds are passed in, constrain the view of the animation to the region bounds.
        layout_kwargs = self._create_layout_with_region_bounds()
        if layout_kwargs:
            fig.update_layout(**layout_kwargs)

        # Hover Menu
        fig.update_layout({"hoverlabel": {"bgcolor": "white", "font": {"color": "black", "size": 12}}, "hovermode": "closest"})

        fig = self.create_play_and_pause_buttons(fig, redraw=redraw)
        # Time slider: "steps" are populated below
        sliders_dict = self.initial_sliders_dict()
        stored_figures = []
        frames = []

        # #####################################################################
        # Deserialize platform steps & Plot
        # #####################################################################
        # Store the platform steps to construct a table to return
        platform_steps: dict = {}
        # The step number is the step within the simulation. This starts at step 1
        # and excludes the initial state (e.g. step 0). The `idx` or `step_idx`
        # corresponds to the index of the platform steps. These two values are
        # off by 1.
        step_num = 1
        for step_data in self.progress_bar_func(self.config.episode_artifact.steps):
            step_idx = step_num - 1
            subplot = make_subplots(**subplot_kwargs)

            # Steps start at 1 because the initial state is not stored
            step_blob = {}
            for serialized_platform_data in step_data.platforms:
                position = self.platform_deserializer.get_position(serialized_platform_data)
                position_units = self.platform_deserializer.get_position_units(serialized_platform_data)
                platform_name = self.platform_deserializer.get_platform_name(serialized_platform_data)
                metadata = self.platform_deserializer.get_metadata_for_text_display(serialized_platform_data)
                filtered_metadata = {}
                if selected_metadata:
                    filtered_metadata = {key: value for key, value in metadata.items() if key in selected_metadata}
                step_blob = self._check_and_parse_position(position, position_units)
                step_blob["step"] = step_num
                step_blob.update(metadata)

                if platform_name in platform_steps:
                    platform_steps[platform_name].append(step_blob)
                else:
                    platform_steps[platform_name] = [step_blob]

                position_units_as_string = ",".join(position_units)
                position_as_string = ",".join([str(round(pos, 2)) for pos in position])
                hover_template = f"""
                    <b>Platform: {platform_name}</b><br>
                    Position [{position_units_as_string}]: ({position_as_string})<br>
                    """

                # Draws a point representative of each platform
                platform_point = draw_point(
                    input_coordinates=tuple(position),
                    marker_color=self.config.platform_color_map[platform_name] if self.config.platform_color_map else "black",
                    plot_text=generate_overhead_text(filtered_metadata),
                    text_position="top center",
                    hover_template=hover_template,
                    name=platform_name,
                )
                subplot.add_trace(platform_point, **trajectory_position)
                # Initialize the first frame, which is stored separately from the subplots
                if step_num == 1:
                    fig.add_trace(trace=platform_point, **trajectory_position)

            custom_polygons = self._draw_custom_polygons(step_idx)
            if custom_polygons is not None:
                for polygon in custom_polygons:
                    if step_num == 1:
                        fig.add_trace(trace=polygon, **trajectory_position)
                    subplot.add_trace(trace=polygon, **trajectory_position)

            subplot.update_layout(**layout_kwargs)

            stored_figures.append(subplot)
            # NOTE: the name of the frame and the slider step must match
            frames.append(go.Frame(name=step_num, data=subplot.data, layout=subplot.layout))
            # Add the slider step
            sliders_dict["steps"].append(self.one_slider_step_animation(step=step_num, redraw=redraw))
            step_num += 1
        platform_step_tables = {}
        for platform_name, platform_data in platform_steps.items():
            platform_step_tables[platform_name] = pd.DataFrame(platform_data)
        fig["layout"]["sliders"] = [sliders_dict]
        fig["frames"] = frames
        return fig, stored_figures, platform_step_tables

    def _draw_custom_polygons(self, _) -> list[go.Scatter | go.Scatter3d] | None:  # noqa: PLR6301
        """This is an optional method to add additional shapes to
        each frame of the animation. Override this method to generate
        custom polygons to plot. There is a helper function `draw_custom_polygon`
        that may help with this."""
        return None

    def _create_layout_with_region_bounds(self):
        """Generates the layout keyword arguments for Figure"""
        layout_kwargs = {}
        if self.frame_axis_bounds:
            # #################################################################
            # Layout for 2D ScatterPlot
            # #################################################################
            layout_kwargs["xaxis"] = {"range": self.frame_axis_bounds.xbounds, "autorange": False}
            layout_kwargs["xaxis_title"] = "x-Position"
            # The scaleanchor sets the range of the y-axis to change with the range of the x-axis
            # such that the scale of pixels per unit is in a constant ratio. Both axes are still zoomable
            # but when you zoom into one, the other will zoom the same amount, keeping a fixed midpoint.
            # The scaleratio determines the pixel to unit scale ratio (e.g. 1:1 with x-axis)
            layout_kwargs["yaxis"] = {"range": self.frame_axis_bounds.ybounds, "autorange": False, "scaleanchor": "x", "scaleratio": 1}
            layout_kwargs["yaxis_title"] = "y-Position"
            # #################################################################
            # Layout for 3D ScatterPlot - "Scene" layout
            # #################################################################
            if self.frame_axis_bounds.zbounds:
                # These options are not available for the 3D plot layouts
                del layout_kwargs["yaxis"]["scaleanchor"]
                del layout_kwargs["yaxis"]["scaleratio"]
                layout_kwargs = {"scene": layout_kwargs}
                layout_kwargs["scene"]["zaxis"] = {"range": self.frame_axis_bounds.ybounds, "autorange": False}
                layout_kwargs["scene"]["zaxis_title"] = "z-Position"
                # Locks the aspect ratio so that the animation doesn't change between frames.
                layout_kwargs["scene"]["aspectratio"] = {"x": 1, "y": 1, "z": 1}
        return layout_kwargs

    def one_slider_step_animation(self, step: Any, redraw=False):
        """Returns a dictionary for one slider step for animations.
        This is one element in the `sliders_dict["steps"]` list.

        Parameters
        ----------
        step : Any
            The attribute in the data that defines a step.
        frame_duration_ms : int, optional
            The time in milliseconds of each frame, by default 30
        transition_duration_ms : int, optional
            The time in milliseconds of the transition of each frame, by default 30

        Returns
        -------
        dict
            A dict for one slider step.
        """
        return {
            "args": [
                [step],
                {
                    "frame": {"duration": self._frame_duration_ms, "redraw": redraw},
                    "mode": "immediate",
                    "transition": {"duration": self._transition_duration_ms},
                },
            ],
            "label": step,
            "method": "animate",
        }

    def create_play_and_pause_buttons(self, fig: go.Figure, redraw=False):
        """Updates a plotly `fig_dict` with the parameters for creating play and pause buttons

        Parameters
        ----------
        fig : go.Figure
            Figure option

        Returns
        -------
        Dict:
            The original fig_dict with the `fig_dict["layout"]["updatemenus"]` property
            updated to include buttons.
        """

        # Append the play and pause buttons to it
        update_menus = []
        update_menus.append(
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": self._frame_duration_ms, "redraw": redraw},
                                "fromcurrent": True,
                                "transition": self._transition_duration_ms,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": redraw}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],  # Button Placement
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        )
        fig.update_layout(updatemenus=update_menus)
        return fig

    def initial_sliders_dict(self, step_descriptor: str = "Step"):
        """Initializes the properties for a slider bar. The `step` key must still be filled out

        Parameters
        ----------
        transition_duration_ms : int
            The transition time in seconds for each frame.
        step_descriptor : str, optional
            A string that describes each step (years, step, etc.), by default "Step"

        Returns
        -------
        dict
            A dictionary with the slider properties initializes.
        """
        return {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {"font": {"size": 12}, "prefix": f"{step_descriptor}: ", "visible": True, "xanchor": "right"},
            "transition": {"duration": self._transition_duration_ms},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [],
        }


def draw_custom_polygon(
    x_coordinates: np.ndarray | (list | tuple),
    y_coordinates: np.ndarray | (list | tuple),
    z_coordinates: np.ndarray | (list | tuple) | None = None,
    color: str = "black",
    fill: bool = True,
    name: str | None = None,
) -> go.Scatter | go.Scatter3d:
    """Draws a polygon according to the coordinates provided.

    Parameters
    ----------
    x_coordinates : Union[np.ndarray, List, Tuple]
        An array of x-coordinates that define the polygon
    y_coordinates : Union[np.ndarray, List, Tuple]
        An array of y-coordinates that define the polygon
    color : str, optional
        Color of the shape, by default "black"
    fill : bool, optional
        When True, fills the shape , by default True
    name : Union[str, None], optional
        The name of the trace, will show in the hover menu and legend, by default None

    Returns
    -------
    plotly.graph_objs.Scatter
        Returns a plotly scatter graph object

    Examples
    --------
    >>> wedge_array = matplotlib_wedge_as_array(center=(0,0), radius=10, start_angle_deg=0, stop_angle_deg=10)
    >>> wedge = draw_custom_polygon(x_coordinates=wedge_array[:,0], y_coordinates=wedge_array[:,1], name="wedge")
    >>> fig = go.Figure(wedge)
    >>> fig.show()
    """
    fill_value = "toself" if fill else None
    if z_coordinates:
        polygon = go.Scatter3d(
            x=x_coordinates, y=y_coordinates, z=z_coordinates, line=dict(color=color), fill=fill_value, name=name, mode="lines"
        )
    else:
        polygon = go.Scatter(x=x_coordinates, y=y_coordinates, line=dict(color=color), fill=fill_value, name=name, mode="lines")

    return polygon


def draw_point(
    input_coordinates: tuple,
    marker_color: str = "black",
    marker_size: int | None = None,
    plot_text: str | None = None,
    hover_template: str | None = None,
    name: str | None = None,
    text_position: str | None = "top center",
) -> go.Scatter | go.Scatter3d:
    """Draws a dot.

    Parameters
    ----------
    input_coordinates : Tuple
        The position of the dot (2D) or sphere (3D).
    marker_color : str, optional
        The color of the marker, by default "black"
    plot_text : Union[str, None], optional
        The text that will go above the the dot, by default None
    hover_template : Union[str, None], optional
        The hover template., by default None
    name : Union[str, None], optional
        Name for legend, by default None
    text_position : Union[str, None], optional
        Position of text, by default "top center"

    Returns
    -------
    plotly.graph_objs.Scatter
        Returns a plotly scatter graph object

    Examples
    --------
    >>> dot = draw_point(xy=(10,10), plot_text="a dot")
    >>> fig = go.Figure(dot)
    >>> fig.show()
    """
    assert len(input_coordinates) in [1, 2, 3], "Input coordinates must either be 1- 2- or 3-dimensions"
    mode = "markers"

    if plot_text:
        mode = f"{mode}+text"
    if len(input_coordinates) < 3:
        dot = go.Scatter(
            x=[input_coordinates[0]],
            y=[0] if len(input_coordinates) == 1 else [input_coordinates[1]],
            marker={"color": marker_color, "size": marker_size},
            mode=mode,
            name=name,
            text=plot_text,
            textposition=text_position,
            hovertemplate=hover_template,
        )
    else:
        dot = go.Scatter3d(
            x=[input_coordinates[0]],
            y=[input_coordinates[1]],
            z=[input_coordinates[2]],
            marker={"color": marker_color, "size": marker_size},
            mode=mode,
            name=name,
            text=plot_text,
            textposition=text_position,
            hovertemplate=hover_template,
        )

    return dot


def generate_overhead_text(metadata: dict[str, int | float]) -> str | None:
    """Generates a block of text, separated by new lines. Numeric values will
    be rounded (4-digits)

    Parameters
    ----------
    data : Dict
    Returns
    -------
    str
        Generated block of text, joined together by a newline.
    """
    if len(metadata) == 0:
        return None

    text_block = []
    for name, value in metadata.items():
        if isinstance(value, str | int):
            text_line = f"{name}: {value}"
            text_block.append(text_line)
        elif isinstance(value, float):
            text_line = f"{name}: {value: .4f}"
            text_block.append(text_line)

    # Only return the text block as a string if text_block is not empty
    return "<br>".join(text_block)
