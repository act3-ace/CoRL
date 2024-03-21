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

from corl.simulators.base_parts import BaseController


class BaseRollPitchYawSpeedController(BaseController, abc.ABC):
    """
    BaseRollPitchYawSpeedController abstraction of controlling an aircraft
    """


class BaseSpeedController(BaseController, abc.ABC):
    """[summary]

    Arguments:
        BaseController: [description]
        abc: [description]

    Returns:
        [type] -- [description]
    """


class BaseRollController(BaseController, abc.ABC):
    """[summary]

    Arguments:
        BaseController: [description]
        abc: [description]

    Returns:
        [type] -- [description]
    """


class BasePitchController(BaseController, abc.ABC):
    """[summary]

    Arguments:
        BaseController: [description]
        abc: [description]

    Returns:
        [type] -- [description]
    """


class BaseYawController(BaseController, abc.ABC):
    """[summary]

    Arguments:
        BaseController: [description]
        abc: [description]

    Returns:
        [type] -- [description]
    """


class BasePitchRollController(BaseController, abc.ABC):
    """[summary]

    Arguments:
        BaseController: [description]
        abc: [description]

    Returns:
        [type] -- [description]
    """
