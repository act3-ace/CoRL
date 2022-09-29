"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Common Platform Utils Module
"""
import typing

from corl.libraries.state_dict import StateDict
from corl.simulators.base_parts import BaseController, BasePlatformPart, BaseSensor
from corl.simulators.base_platform import BasePlatform


def get_platform_by_name(state: StateDict, platform_name: str, allow_invalid=False) -> typing.Optional[BasePlatform]:
    """
    Gets a platform from a sim state based on a given agent id (name)

    Parameters
    ----------
    state: StateDict
        State of current platforms in a sim step
    platform_name: str
        name of a given platform to search for in the sim state

    Returns
    -------
    platform: BasePlatform
        The platform with the given platform_name name
    """
    platform: typing.Optional[BasePlatform] = None
    if "_" in platform_name:
        temp = platform_name.split("_", 1)[0]
    else:
        temp = platform_name

    for plat in state.sim_platforms:
        if plat.name == temp:
            platform = plat

    if not allow_invalid and (platform is None or not issubclass(platform.__class__, BasePlatform)):
        raise ValueError(f"Could not find a platform named {platform_name} of class BasePlatform")

    return platform


def get_controller_by_name(platform: BasePlatform, name: str) -> BaseController:
    """
    Gets a platform controller from a platfrom given the controller name

    Parameters
    ----------
    platform: BasePlatform
        platfrom to find part
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
    Gets a platform sensor from a platfrom given the sensor name

    Parameters
    ----------
    platform: BasePlatform
        platfrom to find part
    name: str
        name of a given sensor to search for in the platform

    Returns
    -------
    platform_sensor: BaseSensor
        The platform sensor with the given name
    """
    return get_part_by_name(platform, name, BaseSensor)


T = typing.TypeVar("T", bound=BasePlatformPart)


def get_part_by_name(platform: BasePlatform, name: str, part_type: typing.Type[T] = None) -> T:
    """
    Gets a platform part from a platfrom given the part name

    Parameters
    ----------
    platform: BasePlatform
        platfrom to find part
    name: str
        name of a given part  to search for in the platform

    Returns
    -------
    platform_part: BasePlatformPart
        The platform part with the given name
    """
    if part_type:

        def part_filter(part):
            return isinstance(part, part_type)

        all_parts = filter(part_filter, platform.controllers + platform.sensors)
    else:
        all_parts = platform.controllers + platform.sensors

    for part in all_parts:
        if part.name == name:
            return part
    raise RuntimeError(f"An attached part associated with {name} group could not be found")


def is_platform_operable(state: StateDict, platform_name: str) -> bool:
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
