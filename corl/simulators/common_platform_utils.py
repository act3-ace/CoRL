"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------

Common Platform Utils Module
"""
import typing

from corl.simulators.base_parts import BaseController, BasePlatformPart, BaseSensor
from corl.simulators.base_platform import BasePlatform
from corl.simulators.base_simulator import BaseSimulatorState


def get_platform_by_name(state: BaseSimulatorState, platform_name: str, allow_invalid=False) -> BasePlatform | None:
    """
    Gets a platform from a sim state based on a given agent id (name)

    Parameters
    ----------
    state: BaseSimulatorState
        State of current platforms in a sim step
    platform_name: str
        name of a given platform to search for in the sim state

    Returns
    -------
    platform: BasePlatform
        The platform with the given platform_name name
    """
    platform: BasePlatform | None = None
    platform = state.sim_platforms.get(platform_name, None)

    if not allow_invalid and (platform is None or not issubclass(platform.__class__, BasePlatform)):
        raise ValueError(f"Could not find a platform named {platform_name} of class BasePlatform")

    return platform


def get_controller_by_name(platform: BasePlatform, name: str) -> BaseController:
    """
    Gets a platform controller from a platform given the controller name

    Parameters
    ----------
    platform: BasePlatform
        platform to find part
    name: str
        name of a given controller to search for in the platform

    Returns
    -------
    platform_controller: BaseController
        The platform controller with the given name
    """
    return get_part_by_name(platform, name, BaseController)


def get_sensor_by_name(platform: BasePlatform, name: str) -> BaseSensor:
    """
    Gets a platform sensor from a platform given the sensor name

    Parameters
    ----------
    platform: BasePlatform
        platform to find part
    name: str
        name of a given sensor to search for in the platform

    Returns
    -------
    platform_sensor: BaseSensor
        The platform sensor with the given name
    """
    return get_part_by_name(platform, name, BaseSensor)


T = typing.TypeVar("T", bound=BasePlatformPart)


def get_part_by_name(platform: BasePlatform, name: str, part_type: type[T] | None = None) -> T:
    """
    Gets a platform part from a platform given the part name

    Parameters
    ----------
    platform: BasePlatform
        platform to find part
    name: str
        name of a given part  to search for in the platform

    Returns
    -------
    platform_part: BasePlatformPart
        The platform part with the given name
    """

    found_part = platform.controllers.get(name, None)
    if not found_part:
        found_part = platform.sensors.get(name, None)

    if not found_part:
        raise RuntimeError(f"An attached part associated with {name} group could not be found")
    if part_type and not isinstance(found_part, part_type):
        raise RuntimeError(
            f"An attached part associated with {name} group was found, however get_part_by_name "
            f"was requested to find a {part_type} and the found part {found_part} does not subclass that type"
        )

    return found_part


def is_platform_operable(state: BaseSimulatorState, platform_name: str) -> bool:
    """
    Check if a platform specified by name is operable

    Parameters
    ----------
    state:
        Simulation state
    platform_name:
        Name of the platform checking to be alive
    Returns
        Is platform operable?
    -------
    """

    platform = get_platform_by_name(state, platform_name, allow_invalid=True)
    return platform is not None and platform.operable
