"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------

Environment Common Module
"""


class EnvCommonValues:
    """Environment Common Values"""

    METERS_TO_FEET = 3.28084
    FEET_TO_METERS = 0.3048
    MACH_TO_METERS_PER_SECOND = 343
    NAUTICAL_MILE_TO_METERS = 1852
    NAUTICAL_MILE_TO_KILOMETERS = 1.852
    METERS_TO_NAUTICAL_MILE = 0.000539957
    DEG_TO_RAD = 0.01745329251994329576923
    RAD_TO_DEG = 57.29577951308232087721
    METERS_PER_SECOND_TO_KNOTS = 1.94384
    FEET_PER_SECOND_TO_KNOTS = 0.592484
    KNOTS_TO_METERS_PER_SECOND = 0.514444
    METERS_TO_KILOMETERS = 0.001


class EnvCommonStrings:
    """Environment Common Strings"""

    SLANT_RANGE_RATE = "range_change_rate"
    SLANT_RANGE = "range"
    ASPECT_ANGLE = "aspect_angle"
    CROSS_ANGLE = "heading_cross_angle"
    RELATIVE_ELEVATION = "relative_elevation"

    ALT_ERROR = "alt_error"
    SUM_ALT_ERROR = "sum_alt_error"
    ALT_ERROR_DIFF = "alt_error_diff"
