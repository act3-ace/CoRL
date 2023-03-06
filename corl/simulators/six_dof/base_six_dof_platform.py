"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Abstaction Class for 6DOF platform types that can provide
some properties for common usage
"""
import abc

import numpy as np

import corl.simulators.six_dof.base_six_dof_properties as six_dof_props
from corl.simulators.base_platform import BasePlatform


class Base6DOFPlatform(BasePlatform):
    """
    Further abstraction of BasePlatform, this adds some common
    6dof properties that may be useful for dealing with simulators
    using aircraft
    """

    @property
    @abc.abstractmethod
    def position(self) -> np.ndarray:
        """
        The position of the object in space. Details on the position format are provided by position_properties

        Returns
        -------
        np.ndarray
            The position of the object in space
        """

    position_properties = six_dof_props.LatLonAltProp()

    @property
    @abc.abstractmethod
    def orientation(self) -> np.ndarray:
        """
        The orientation of the platform. For orientation formatting see orientation_properties

        Returns
        -------
        np.ndarray
            The orientation of the platform
        """

    orientation_properties = six_dof_props.OrientationProp()

    @property
    @abc.abstractmethod
    def velocity_ned(self) -> np.ndarray:
        """get the velocity in true airspeed NED (m/s)

        Returns:
            np.ndarray -- The velocity in true airspeed NED (m/s)
        """

    velocity_ned_properties = six_dof_props.VelocityNEDProp()
    angular_velocity_properties = six_dof_props.VelocityNEDProp(
        description="angular velocity for yaw rate, pitch rate and roll rate respectively"
    )

    @property
    @abc.abstractmethod
    def acceleration_ned(self) -> np.ndarray:
        """gets the acceleration in the NED

        Returns:
            np.ndarray -- The acceleration in the NED
        """

    acceleration_ned_properties = six_dof_props.AccelerationNEDProp()

    @property
    @abc.abstractmethod
    def speed(self) -> np.ndarray:
        """Get the speed of the platform

        Returns:
            np.ndarray -- The true airspeed of the platform in m/s
        """

    speed_properties = six_dof_props.TrueAirSpeedProp(name="speed", high=[1700.0], unit=["mpstas"], description="true airspeed in m/s")
