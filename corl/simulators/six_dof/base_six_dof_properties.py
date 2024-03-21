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
from typing import Annotated

import numpy as np
from pydantic import Field, StrictFloat, StrictStr

from corl.libraries.property import BoxProp


class LatLonProp(BoxProp):
    """
    Lat Lon space
    """

    name: str = "LatLon"
    low: Annotated[list[StrictFloat], Field(min_length=2, max_length=2)] = [-90.0, -180.0]
    high: Annotated[list[StrictFloat], Field(min_length=2, max_length=2)] = [90.0, 180.0]
    unit: StrictStr = "degree"
    description: str = "Lat Lon"


class AltitudePropMeters(BoxProp):
    """
    Altitude space meters
    """

    name: str = "altitude"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    unit: StrictStr = "meter"
    description: str = "true altitude of the platform in meters. Altitude is measured above WGS-84 ellipsoid"


class AltitudeFeetProp(BoxProp):
    """
    Altitude space ft
    """

    name: str = "altitude"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    unit: StrictStr = "foot"
    description: str = "true altitude of the platform in ft"


class AltitudeMeterProp(BoxProp):
    """
    Altitude space meters ft
    """

    name: str = "altitude_m"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    unit: StrictStr = "meter"
    description: str = "true altitude of the platform in m"


class AltitudeRateProp(BoxProp):
    """
    Altitude rate space
    """

    name: str = "altitude_rate"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    unit: StrictStr
    description: str = "true altitude rate of the platform"


class OrientationProp(BoxProp):
    """
    Orientation space
    """

    name: str = "orientation"
    low: Annotated[list[StrictFloat], Field(min_length=3, max_length=3)] = [-np.pi] * 3
    high: Annotated[list[StrictFloat], Field(min_length=3, max_length=3)] = [np.pi] * 3
    unit: StrictStr = "rad"
    description: str = "yaw/heading, pitch, roll. Orientation is relative to the NED frame"


class OrientationRateProp(BoxProp):
    """
    Orientation rate space
    """

    name: str = "orientation_rate"
    low: Annotated[list[StrictFloat], Field(min_length=3, max_length=3)] = [-3 * np.pi] * 3
    high: Annotated[list[StrictFloat], Field(min_length=3, max_length=3)] = [3 * np.pi] * 3
    unit: StrictStr = "rad/s"
    description: str = "yaw rate, pitch rate, roll rate"


class FuelProp(BoxProp):
    """
    Fuel space in fuel / total fuel for the platform
    """

    name: str = "fuel_percentage"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [1.0]
    unit: StrictStr = "dimensionless"
    description: str = "ratio of remaining fuel / total fuel, 0 is empty and 1 is full"


class MachProp(BoxProp):
    """
    Mach speed space
    """

    name: str = "speed_mach"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [1.5]
    unit: StrictStr = "dimensionless"
    description: str = "speed of the platform in Mach"


class KcasProp(BoxProp):
    """
    Kcas speed space
    """

    name: str = "speed_kcas"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)]
    unit: StrictStr = "kcas"
    description: str = "KCAS (in knots)"


class KtasProp(BoxProp):
    """
    Ktas speed space
    """

    name: str = "speed_ktas"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)]
    unit: StrictStr = "knot"
    description: str = "KTAS (in knots)"


class KiasProp(BoxProp):
    """
    Kias speed space
    """

    name: str = "speed_kias"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)]
    unit: StrictStr = "kias"
    description: str = "KIAS (in knots)"


class TrueAirSpeedProp(BoxProp):
    """
    True airspeed space
    """

    name: str = "true_air_speed_fts"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)]
    unit: StrictStr = "foot / second"
    description: str = "true airspeed of the platform in foot per second"


class VelocityNEDProp(BoxProp):
    """
    Velocity NED space
    """

    name: str = "velocity_ned"
    low: Annotated[list[StrictFloat], Field(min_length=3, max_length=3)] = [-3000.0] * 3
    high: Annotated[list[StrictFloat], Field(min_length=3, max_length=3)] = [3000.0] * 3
    unit: StrictStr = "meter / second"
    description: str = "velocity in true airspeed (m/s) along north, east, down axis"


class AccelerationNEDProp(BoxProp):
    """
    Acceleration NED space
    """

    name: str = "acceleration_ned"
    low: Annotated[list[StrictFloat], Field(min_length=3, max_length=3)] = [-3000.0] * 3
    high: Annotated[list[StrictFloat], Field(min_length=3, max_length=3)] = [3000.0] * 3
    unit: StrictStr = "m/s^2"
    description: str = "acceleration in true airspeed (m/s^2) along north, east, down axis"


class FlightPathAngleProp(BoxProp):
    """
    Flight path angle space in degrees
    """

    name: str = "flight_path_angle_deg"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-180.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [180.0]
    unit: StrictStr = "degree"
    description: str = "flight path angle of the platform in degrees"


class FlightPathAngleRadProp(BoxProp):
    """
    Flight path angle in radians
    """

    name: str = "flight_path_angle_rad"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-np.pi / 2]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [np.pi / 2]
    unit: StrictStr = "rad"
    description: str = "flight path angle of the platform in rad"


class GLoadProp(BoxProp):
    """
    Gload space
    """

    name: str = "g_load"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-20.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [20.0]
    unit: StrictStr = "standard_gravity"
    description: str = "G load on the platform"


class GloadNzProp(GLoadProp):
    """
    Gload space in z
    """

    name: str = "g_load NZ"


class GloadNyProp(GLoadProp):
    """
    Gload space in y
    """

    name: str = "g_load NY"


class GloadNxProp(GLoadProp):
    """
    Gload space in x
    """

    name: str = "g_load NX"


class AngleOfAttackProp(BoxProp):
    """
    Angle of attack space
    """

    name: str = "angle_of_attack"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-180.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [180.0]
    unit: StrictStr = "degree"
    description: str = "The angle of attack of the platform in degrees"


class AngleOfAttackRateProp(BoxProp):
    """
    Angle of attack rate space
    """

    name: str = "angle_of_attack_rate"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-180.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [180.0]
    unit: StrictStr = "deg/s"
    description: str = "The angle of attack rate of the platform in degrees/s"


class AngleOfSideSlipProp(BoxProp):
    """
    Angle of slide slip
    """

    name: str = "angle_of_side_slip"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-90.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [90.0]
    unit: StrictStr = "degree"
    description: str = "The angle of side slip of the platform in degrees"


class AngleOfSideSlipRateProp(BoxProp):
    """
    Angle of slide slip rate space
    """

    name: str = "angle_of_side_slip_rate_dps"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-180.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [180.0]
    unit: StrictStr = "deg/s"
    description: str = "The angle of side slip rate of the platform in degrees/sec"


class FuelWeightProp(BoxProp):
    """
    Fuel weight space
    """

    name: str = "fuel_weight_lbs"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [10000.0]
    unit: StrictStr = "lbs"
    description: str = "Fuel weight"


class WindDirectionProp(BoxProp):
    """
    Wind direction space
    """

    # https://docs.google.com/spreadsheets/d/1L7D4uqVQzY7rODqtnumB0Kv0-veQ1bHIItLWWrpOOmA/edit#gid=0
    name: str = "wind_direction_deg"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [360.0]
    unit: StrictStr = "degree"
    description: str = "Wind Direction in Degrees of the Platform"


class WindSpeedProp(BoxProp):
    """
    Wind speed space
    """

    # https://docs.google.com/spreadsheets/d/1L7D4uqVQzY7rODqtnumB0Kv0-veQ1bHIItLWWrpOOmA/edit#gid=0
    name: str = "wind_speed_kts"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [200.0]
    unit: StrictStr = "kts"
    description: str = "Wind speed in kts of the Platform"


class YawProp(BoxProp):
    """
    Yaw space
    """

    name: str = "orientation_yaw"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-180.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [180.0]
    unit: StrictStr = "degree"
    description: str = "Get RPM Yaw (deg)"


class PitchProp(BoxProp):
    """
    Pitch space
    """

    name: str = "orientation_pitch"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-180.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [180.0]
    unit: StrictStr = "degree"
    description: str = "Get RPM Pitch (deg)"


class RollProp(BoxProp):
    """
    Roll space
    """

    name: str = "orientation_roll"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-180.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [180.0]
    unit: StrictStr = "degree"
    description: str = "Get RPM Roll (deg)"


class YawRateProp(BoxProp):
    """
    Yaw rate space
    """

    name: str = "orientation_yaw_rate"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-3 * 180.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [3 * 180.0]
    unit: StrictStr = "deg/s"
    description: str = "Get RPM yaw rate (deg/sec)"


class PitchRateProp(BoxProp):
    """
    Pitch rate space
    """

    name: str = "orientation_pitch_rate"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-3 * 180.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [3 * 180.0]
    unit: StrictStr = "deg/s"
    description: str = "Get RPM pitch rate (deg/sec)"


class RollRateProp(BoxProp):
    """
    Roll rate space
    """

    name: str = "orientation_roll_rate"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-540.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [540.0]
    unit: StrictStr = "deg/s"
    description: str = "Get RPM roll rate (deg/sec)"


class HeadingProp(BoxProp):
    """
    default heading space
    """

    name: str = "heading"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-180.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [180.0]
    unit: StrictStr = "degree"
    description: str = "Direct (in degrees) the target heading"


def property_factor(prefix_name, input_property_class, low_prop=None, high_prop=None) -> type:
    """
    factory for creating space mods...
    """
    if issubclass(input_property_class, BoxProp):
        if low_prop is not None and high_prop is not None:

            class NewProperty(input_property_class):
                """
                dynamic class creation
                """

                low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = low_prop
                high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = high_prop

            return type(prefix_name + input_property_class.__name__, (NewProperty,), {})

        if low_prop is not None:

            class NewProperty(input_property_class):  # type: ignore[no-redef]
                """
                dynamic class creation
                """

                low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = low_prop

            return type(prefix_name + input_property_class.__name__, (NewProperty,), {})

        if high_prop is not None:

            class NewProperty(input_property_class):  # type: ignore[no-redef]
                """
                dynamic class creation
                """

                high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = high_prop

            return type(prefix_name + input_property_class.__name__, (NewProperty,), {})

        class NewProperty(input_property_class):  # type: ignore[no-redef]
            """
            dynamic class creation
            """

        return type(prefix_name + input_property_class.__name__, (NewProperty,), {})

    raise NotImplementedError(f"Did not implement for target base type --- {input_property_class.__bases__}")
