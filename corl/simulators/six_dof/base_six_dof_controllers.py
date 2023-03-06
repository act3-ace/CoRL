"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Base Integration Controls Module
"""
import abc
import typing

from corl.simulators.base_parts import BaseController


class BaseRollPitchYawSpeedController(BaseController, abc.ABC):
    """
    BaseRollPitchYawSpeedController abstraction of controlling an aircraft
    """

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return {"roll_control", "pitch_control", "yaw_control", "speed_control"}


class BaseSpeedController(BaseController, abc.ABC):
    """[summary]

    Arguments:
        BaseController: [description]
        abc: [description]

    Returns:
        [type] -- [description]
    """

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return {"speed_control"}


class BaseRollController(BaseController, abc.ABC):
    """[summary]

    Arguments:
        BaseController: [description]
        abc: [description]

    Returns:
        [type] -- [description]
    """

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return {"roll_control"}


class BasePitchController(BaseController, abc.ABC):
    """[summary]

    Arguments:
        BaseController: [description]
        abc: [description]

    Returns:
        [type] -- [description]
    """

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return {"pitch_control"}


class BaseYawController(BaseController, abc.ABC):
    """[summary]

    Arguments:
        BaseController: [description]
        abc: [description]

    Returns:
        [type] -- [description]
    """

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return {"yaw_control"}


class BasePitchRollController(BaseController, abc.ABC):
    """[summary]

    Arguments:
        BaseController: [description]
        abc: [description]

    Returns:
        [type] -- [description]
    """

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return {"pitch_control", "roll_control"}
