"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from abc import ABC, abstractmethod


class PlatformDeserializerPlotlyAnimation(ABC):
    """Abstract class to define how to deserialize a platform for plotting"""

    @abstractmethod
    def get_position(self, serialized_data: dict) -> tuple | list:
        """Deserialized position data. This tells plotly where to plot
        a point for each platform on one step.

        Parameters
        ----------
        serialized_data : Dict
            Serialized platform data.

        Returns
        -------
        Union[Tuple, List]
            Position for each step, stored as a list or tuple.
            2-points are expected for 2-dimensional day and 3-points
            are expected for 3-dimensional data.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dimensionality(self, serialized_data: dict) -> int:
        """Returns the dimensionality of the world - expected to be
        2D or 3D

        Parameters
        ----------
        serialized_data : Dict
            _description_

        Returns
        -------
        int
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    @abstractmethod
    def get_position_units(self, serialized_data: dict) -> tuple | list:
        """Deserialized position units. This tells plotly what the units of
        the position are. The units are shown in the animation.

        Parameters
        ----------
        serialized_data : Dict
            Serialized platform data.

        Returns
        -------
        Union[Tuple,List]
            A tuple of units.
        """
        raise NotImplementedError

    @abstractmethod
    def get_platform_name(self, serialized_data: dict) -> str:
        """Returns the platform name

        Parameters
        ----------
        serialized_data : Dict
            Serialized platform data.

        Returns
        -------
        str
            Platform name
        """
        raise NotImplementedError

    @abstractmethod
    def get_metadata_for_text_display(self, serialized_data) -> dict[str, int | (float | str)]:
        """Deserializes metadata into a flattened dictionary. This is anything
        you want to include as optional text to displayed above the platform.

        Parameters
        ----------
        serialized_data : Dict
            Serialized platform data.

        Returns
        -------
        Dict[str, Union[int, float, str]]
            Any information you want to include in a flattened dictionary
            that could be displayed above the platform. This could be the velocity,
            heading, applied controls, sensors, etc.
        """
        raise NotImplementedError
