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


class serialize_Docking_1d(PlatformSerializer):
    """
    This is a implementation of the abstract class PlatformSerializer.
    This provides functionality for serializing a 1D docking platform.
    """

    def serialize(self, platform):
        """
        Parameters
        ----------
        platform
            The incoming 1d docking platform object that needs to be serialized

        Returns
        -------
        dict
            the serialized represnetation of the 1d docking platform.
        """
        dictionary = {
            'name': platform.name,
            'position': platform.position,
            'velocity': platform.velocity,
            'sim_time': platform.sim_time,
            'operable': platform.operable,
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
    This provides functionality for serializing a Pong Platform.
    """

    def serialize(self, platform):
        """
        Parameters
        ----------
        platform
            The incoming pong platform object that needs to be serialized

        Returns
        -------
        dict
            the serialized represnetation of the pong platform.
        """
        dictionary = {
            'name': platform.name,
            'operable': platform.operable,
        }

        dictionary["controllers"] = {}
        for controller in platform.controllers.values():
            dictionary["controllers"][controller.name] = {"applied_controls": controller.get_applied_control()}

        dictionary["sensors"] = {}
        for sensor in platform.sensors.values():
            dictionary["sensors"][sensor.name] = {"measurement": sensor.get_measurement()}

        return dictionary
