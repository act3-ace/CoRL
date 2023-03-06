# pylint: disable=too-many-lines
"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
property definitions
"""
import typing

import numpy as np
from pydantic import Field, StrictFloat, StrictStr
from typing_extensions import Annotated

from corl.libraries.property import BoxProp


class LatLonProp(BoxProp):
    """
    Lat Lon space
    """
    name: str = "LatLon"
    low: Annotated[typing.List[StrictFloat], Field(min_items=2, max_items=2)] = [-90.0, -180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=2, max_items=2)] = [90.0, 180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=2, max_items=2)] = ["deg", "deg"]
    description: str = "Lat Lon"


class LatLonAltProp(BoxProp):
    """
    Lat Lon Alt space
    """
    name: str = "position"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-90.0, -180.0, 0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [90.0, 180.0, 0.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["deg", "deg", "m"]
    description: str = "Geodetic Latitude, Longitude, Altitude. Altitude is measured above WGS-84 ellipsoid"


class AltitudePropMeters(BoxProp):
    """
    Altitude space meters
    """
    name: str = "altitude"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["meters"]
    description: str = "true altitude of the platform in meters"


class AltitudePropFeet(BoxProp):
    """
    Altitude space meters ft
    """
    name: str = "altitude"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["ft"]
    description: str = "true altitude of the platform in ft"


class AltitudeRateProp(BoxProp):
    """
    Altitude rate space
    """
    name: str = "altitude_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)]
    description: str = "true altitude rate of the platform"


class OrientationProp(BoxProp):
    """
    Orientation space
    """
    name: str = "orientation"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-np.pi] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [np.pi] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["rad"] * 3
    description: str = "yaw/heading, pitch, roll. Orientation is relative to the NED frame"


class OrientationRateProp(BoxProp):
    """
    Orientation rate space
    """
    name: str = "orientation_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-3 * np.pi, -3 * np.pi, -3 * np.pi]
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [3 * np.pi, 3 * np.pi, 3 * np.pi]
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["rad/s"] * 3
    description: str = "yaw rate, pitch rate, roll rate"


class FuelProp(BoxProp):
    """
    Fuel space in fuel / total fuel for the platform
    """
    name: str = "fuel_percentage"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [1.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["fraction"]
    description: str = "ratio of remaining fuel / total fuel, 0 is empty and 1 is full"


class MachProp(BoxProp):
    """
    Mach speed space
    """
    name: str = "speed_mach"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["Ma"]
    description: str = "speed of the platform in Mach"


class KcasProp(BoxProp):
    """
    Kcas speed space
    """
    name: str = "speed_kcas"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["kcas"]
    description: str = "KCAS (in knots)"


class KtasProp(BoxProp):
    """
    Ktas speed space
    """
    name: str = "speed_ktas"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["ktas"]
    description: str = "KTAS (in knots)"


class KiasProp(BoxProp):
    """
    Kias speed space
    """
    name: str = "speed_kias"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["kias"]
    description: str = "KIAS (in knots)"


class TrueAirSpeedProp(BoxProp):
    """
    True airspeed space
    """
    name: str = "true_air_speed_fts"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["fpstas"]
    description: str = "true airspeed of the platform in feet per second"


class VelocityNEDProp(BoxProp):
    """
    Velocity NED space
    """
    name: str = "velocity_ned"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-3000.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [3000.0] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["mpstas"] * 3
    description: str = "velocity in true airspeed (m/s) along north, east, down axis"


class AccelerationNEDProp(BoxProp):
    """
    Acceleration NED space
    """
    name: str = "acceleration_ned"
    low: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [-3000.0] * 3
    high: Annotated[typing.List[StrictFloat], Field(min_items=3, max_items=3)] = [3000.0] * 3
    unit: Annotated[typing.List[StrictStr], Field(min_items=3, max_items=3)] = ["m/s^2"] * 3
    description: str = "accleration in true airspeed (m/s^2) along north, east, down axis"


class FlightPathAngleProp(BoxProp):
    """
    Flight path angle space in degrees
    """
    name: str = "flight_path_angle_deg"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg"]
    description: str = "flight path angle of the platform in degrees"


class FlightPathAngleRadProp(BoxProp):
    """
    Flight path angle in radians
    """
    name: str = "flight_path_angle_rad"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-np.pi / 2]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [np.pi / 2]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["rad"]
    description: str = "flight path angle of the platform in rad"


class GloadProp(BoxProp):
    """
    Gload space
    """
    name: str = "g_load"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-20.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [20.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["G"]
    description: str = "G load on the platform"


class GloadNzProp(GloadProp):
    """
    Gload space in z
    """
    name: str = "g_load NZ"


class GloadNyProp(GloadProp):
    """
    Gload space in y
    """
    name: str = "g_load NY"


class GloadNxProp(GloadProp):
    """
    Gload space in x
    """
    name: str = "g_load NX"


class AngleOfAttackProp(BoxProp):
    """
    Angle of attack space
    """
    name: str = "angle_of_attack"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg"]
    description: str = "The angle of attack of the platform in degrees"


class AngleOfAttackRateProp(BoxProp):
    """
    Angle of attack rate space
    """
    name: str = "angle_of_attack_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg/s"]
    description: str = "The angle of attack rate of the platform in degrees/s"


class AngleOfSideSlipProp(BoxProp):
    """
    Angle of slide slip
    """
    name: str = "angle_of_side_slip"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-90.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [90.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg"]
    description: str = "The angle of side slip of the platform in degrees"


class AngleOfSideSlipRateProp(BoxProp):
    """
    Angle of slide slip rate space
    """
    name: str = "angle_of_side_slip_rate_dps"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg/s"]
    description: str = "The angle of side slip rate of the platform in degrees/sec"


class FuelWeightProp(BoxProp):
    """
    Fuel weight space
    """
    name: str = "fuel_weight_lbs"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [10000.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["lbs"]
    description: str = "Fuel weight"


class WindDirectionProp(BoxProp):
    """
    Wind direction space
    """
    # https://docs.google.com/spreadsheets/d/1L7D4uqVQzY7rODqtnumB0Kv0-veQ1bHIItLWWrpOOmA/edit#gid=0
    name: str = "wind_direction_deg"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [360.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg"]
    description: str = "Wind Direction in Degrees of the Platform"


class WindSpeedProp(BoxProp):
    """
    Wind speed space
    """
    # https://docs.google.com/spreadsheets/d/1L7D4uqVQzY7rODqtnumB0Kv0-veQ1bHIItLWWrpOOmA/edit#gid=0
    name: str = "wind_speed_kts"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [200.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["kts"]
    description: str = "Wind speed in kts of the Platform"


class YawProp(BoxProp):
    """
    Yaw space
    """
    name: str = "orientation_yaw"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg"]
    description: str = "Get RPM Yaw (deg)"


class PitchProp(BoxProp):
    """
    Pitch space
    """
    name: str = "orientation_pitch"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg"]
    description: str = "Get RPM Pitch (deg)"


class RollProp(BoxProp):
    """
    Roll space
    """
    name: str = "orientation_roll"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg"]
    description: str = "Get RPM Roll (deg)"


class YawRateProp(BoxProp):
    """
    Yaw rate space
    """
    name: str = "orientation_yaw_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-3 * 180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [3 * 180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg/s"]
    description: str = "Get RPM yaw rate (deg/sec)"


class PitchRateProp(BoxProp):
    """
    Pitch rate space
    """
    name: str = "orientation_pitch_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-3 * 180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [3 * 180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg/s"]
    description: str = "Get RPM pitch rate (deg/sec)"


class RollRateProp(BoxProp):
    """
    Roll rate space
    """
    name: str = "orientation_roll_rate"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-3 * 180.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [3 * 180.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["deg/s"]
    description: str = "Get RPM roll rate (deg/sec)"
