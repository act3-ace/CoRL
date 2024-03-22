"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Base animator class for streamlit application.
"""
from abc import ABC, abstractmethod
from typing import TypeVar

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.libraries.functor import Functor


class FrameAxisBounds(BaseModel):
    """The frame axis bounds for the animation (min, max)"""

    xbounds: tuple[float | int, float | int]
    ybounds: tuple[float | int, float | int]
    zbounds: tuple[float | int, float | int] | None = None


class Polygon(BaseModel):
    """A shape is defined by the name of the shape and points that
    draw the shape. The z-coordinate should be set to None when the shape is
    2-Dimensional."""

    name: str
    x_coordinates: np.ndarray | (list | tuple)
    y_coordinates: np.ndarray | (list | tuple)
    z_coordinates: np.ndarray | (list | tuple) | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseAnimatorValidator(BaseModel):
    """Validator for the BaseAnimator"""

    episode_artifact: TypeVar(EpisodeArtifact)  # type: ignore
    platform_serializer_map: dict[str, str]
    platform_color_map: dict[str, str] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseAnimator(ABC):
    """Abstract class for creating a streamlit animator."""

    def __init__(
        self, episode_artifact: EpisodeArtifact, platform_serializer_map: dict[str, str], platform_color_map: dict[str, str] | None = None
    ) -> None:
        """
        Parameters
        ----------
        episode_artifact : EpisodeArtifact
            An episode artifact.
        """
        self.config = self.get_validator()(
            episode_artifact=episode_artifact, platform_color_map=platform_color_map, platform_serializer_map=platform_serializer_map
        )
        self.get_platform_color_map_from_platform_name()

    @staticmethod
    def get_validator() -> type[BaseAnimatorValidator]:
        """Get the validator

        Returns
        -------
        Type[BaseAnimatorValidator]
            Validator for BaseAnimator
        """
        return BaseAnimatorValidator

    def get_platform_color_map_from_platform_name(self):
        """Helper function to creates a map that between platform names and a CSS color.
        When the platform name is already in the colormap, skips it.
        """
        platform_names = list(self.config.episode_artifact.platform_to_agents.keys())
        platform_color_map = self.config.platform_color_map
        if platform_color_map is None:
            platform_color_map = {}
        color_map = mcolors.CSS4_COLORS
        for name in platform_names:
            if name in platform_color_map:
                continue
            # Checks if a color is part of the platform name
            possible_color = [color for color_name, color in color_map.items() if color_name in name]
            if possible_color:
                platform_color_map[name] = possible_color[0]
        self.config.platform_color_map = platform_color_map

    def get_deserializer(self):
        """
        Returns the deserializer from the serializer found in the episode artifact.
        Grabs the deserializer from the platform serializer map
        """
        module = self.config.episode_artifact.platform_serializer.__module__
        cls_name = self.config.episode_artifact.platform_serializer.__class__.__name__
        class_name = f"{module}.{cls_name}"
        deserializer = self.config.platform_serializer_map.get(class_name)
        if deserializer is None:
            return None
        return Functor(functor=deserializer).functor

    @abstractmethod
    def get_dimensionality(self) -> int:
        """Returns the dimensionality of the animation

        Returns
        -------
        int
            Dimensionality of the animation world (e.g. 2 or 3)
        """
        raise NotImplementedError

    @abstractmethod
    def set_frame_axis_bounds(self, frame_axis_bounds: FrameAxisBounds):
        """Abstract method for setting the axis bounds"""
        raise NotImplementedError

    @abstractmethod
    def get_metadata_names(self) -> list[str]:
        """Abstract method that returns a list of metadata values that could be
        used to select additional information to plot above the platform.
        Return and empty list to exclude this feature"""

    @abstractmethod
    def generate_animation(self, selected_metadata: list[str] | None = None) -> tuple[go.Figure, list[go.Figure], dict[str, pd.DataFrame]]:
        """Abstract method. Main method to generate the animation"""
