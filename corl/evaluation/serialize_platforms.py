"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
This module contains logic to serialize different platforms
"""


from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer
from corl.simulators.docking_1d.properties import PositionProp as Docking1dPos
from corl.visualization.platform_plotting_deserializer import PlatformDeserializerPlotlyAnimation


class serialize_Docking_1d(PlatformSerializer):
    """
    This is a implementation of the abstract class PlatformSerializer.
    This provides functionality for serializing a 1D docking platform.
    """

    def serialize(self, platform):  # noqa: PLR6301
        """
        Parameters
        ----------
        platform
            The incoming 1d docking platform object that needs to be serialized

        Returns
        -------
        dict
            the serialized representation of the 1d docking platform.
        """
        dictionary = {
            "name": platform.name,
            "position": platform.position,
            "velocity": platform.velocity,
            "sim_time": platform.sim_time,
            "operable": platform.operable,
        }

        dictionary["controllers"] = {}
        for controller in platform.controllers.values():
            dictionary["controllers"][controller.name] = {"applied_controls": controller.get_applied_control()}

        dictionary["sensors"] = {}
        for sensor in platform.sensors.values():
            dictionary["sensors"][sensor.name] = {"measurement": sensor.get_measurement()}

        return dictionary


class Serialize_Pong(PlatformSerializer):
    """
    This is a implementation of the abstract class PlatformSerializer.
    This provides functionality for serializing a Pong Platform to be stored
    in the episode artifact.
    """

    def serialize(self, platform):  # noqa: PLR6301
        """
        Parameters
        ----------
        platform
            The incoming pong platform object that needs to be serialized

        Returns
        -------
        dict
            the serialized representation of the pong platform.
        """
        serialized_data = {
            "name": platform.name,
            "ball_hits": platform.paddle.ball_hits,
            "position": (platform.paddle.x, platform.paddle.y),
            "paddle_height": platform.paddle.height,
            "paddle_width": platform.paddle.width,
            "operable": platform.operable,
            "paddle_type": platform.paddle_type.name,
        }

        serialized_data["controllers"] = {}
        for controller in platform.controllers.values():
            serialized_data["controllers"][controller.name] = {"applied_controls": controller.get_applied_control()}

        serialized_data["sensors"] = {}
        for sensor in platform.sensors.values():
            serialized_data["sensors"][sensor.name] = {"measurement": sensor.get_measurement()}

        return serialized_data


class DeserializeDocking1dPlotlyAnimation(PlatformDeserializerPlotlyAnimation):
    """
    The deserializer for Docking-1d. This is an example
    of how to return the necessary minimal data for
    rendering a plotly trajectory animation within the streamlit application.
    """

    def get_position(self, serialized_data) -> tuple | list:  # noqa: PLR6301
        return serialized_data["position"]

    def get_position_units(self, serialized_data: dict) -> tuple | list:  # noqa: PLR6301
        position_units = (Docking1dPos().unit for _ in serialized_data["position"])
        return tuple(position_units)

    def get_platform_name(self, serialized_data: dict) -> str:  # noqa: PLR6301
        return serialized_data["name"]

    def get_dimensionality(self, serialized_data: dict) -> int:  # noqa: PLR6301
        return len(serialized_data["position"])

    def get_metadata_for_text_display(self, serialized_data) -> dict[str, int | float | str]:  # noqa: PLR6301
        return {"sim_time": serialized_data["sim_time"], "velocity": serialized_data["velocity"].item()}


class DeserializePongPlotlyAnimation(PlatformDeserializerPlotlyAnimation):
    """
    The deserializer for Pong. This is an example
    of how to return the necessary minimal data for
    rendering a plotly trajectory animation within the streamlit application.
    """

    def get_position(self, serialized_data) -> tuple | list:  # noqa: PLR6301
        """Gets the position from the serialized data"""
        return serialized_data["position"]

    def get_position_units(self, serialized_data: dict) -> tuple | list:  # noqa: PLR6301
        """Returns the position units - for Pong, the position units are the coordinates
        of the display screen and are unit-less."""
        position_units = ("None" for _ in serialized_data["position"])
        return tuple(position_units)

    def get_platform_name(self, serialized_data) -> str:  # noqa: PLR6301
        """Returns the name of the platform"""
        return serialized_data["name"]

    def get_dimensionality(self, serialized_data: dict) -> int:  # noqa: PLR6301
        """Returns the dimensionality of the world"""
        return len(serialized_data["position"])

    def get_paddle_dimensions(self, serialized_data: dict):  # noqa: PLR6301
        return {"width": serialized_data["paddle_width"], "height": serialized_data["paddle_height"]}

    def get_metadata_for_text_display(self, serialized_data):  # noqa: PLR6301
        """Returns data in a flattened dictionary that can be optionally displayed
        above each platform"""
        return {
            "ball_hits": serialized_data["ball_hits"],
        }
