"""
This module defines the measurement and control properties for Double Integrator spacecraft sensors and controllers.
"""

from typing import Annotated

from pydantic import Field, StrictFloat

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
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-150.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [150.0]
    unit: str = "meter"
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
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-50.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [50.0]
    unit: str = "meter / second"
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
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [-1.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [1.0]
    unit: str = "newtons"
    description: str = "Direct Thrust Control"
