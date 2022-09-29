"""
This module defines the measurement and control properties for Double Integrator spacecraft sensors and controllers.
"""

import typing

from pydantic import Field, StrictFloat, StrictStr
from typing_extensions import Annotated

from corl.libraries.property import BoxProp


class PositionProp(BoxProp):
    """
    Position sensor properties.

    name : str
        sensor property name
    low : list[float]
        minimum bounds of sensor output
    high : list[float]
        maximum bounds of sensor output
    unit : str
        unit of measurement for sensor output
    description : str
        description of sensor properties
    """

    name: str = "position"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-50000.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [50000.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["m"]
    description: str = "Position Sensor Properties"


class VelocityProp(BoxProp):
    """
    Velocity sensor properties.

    name : str
        sensor property name
    low : list[float]
        minimum bounds of sensor output
    high : list[float]
        maximum bounds of sensor output
    unit : str
        unit of measurement for sensor output
    description : str
        description of sensor properties
    """

    name: str = "velocity"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-2000.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [2000.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["m/s"]
    description: str = "Velocity Sensor Properties"


class ThrustProp(BoxProp):
    """
    Thrust control properties.

    name : str
        control property name
    low : list[float]
        minimum bounds of control input
    high : list[float]
        maximum bounds of control input
    unit : str
        unit of measurement for control input
    description : str
        description of control properties
    """

    name: str = "thrust"
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [-1.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [1.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["newtons"]
    description: str = "Direct Thrust Control"
