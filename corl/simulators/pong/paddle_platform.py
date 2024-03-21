"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
This module extends corl.simulators.base_platform.BasePlatform to create a paddle platform
used to interact with the pong
"""

from enum import Enum

import pygame

from corl.simulators.base_platform import BasePlatform, BasePlatformValidator


class PaddleType(str, Enum):
    """
    Enumeration describing the paddle side [left or right]
    """

    LEFT = "left"
    RIGHT = "right"


class PaddlePlatformValidator(BasePlatformValidator):
    """paddle platform validator"""

    paddle_type: PaddleType


class PaddlePlatform(BasePlatform):
    """
    A platform representing a pong paddles.
    Allows for saving of "moves" to give an action to the pong game
    during the environment step function

    Parameters
    ----------
    platform_name : str
        Name of the platform
    platform : PaddleType
        The paddle side corresponding to what the platform in controlling
        in the pong game
    platform_config : dict
        Platform-specific configuration dictionary
    paddle: paddle
        The specific paddle object from the simulator
    """

    def __init__(self, platform_name, parts_list, paddle_type, paddle) -> None:
        self.config: PaddlePlatformValidator
        super().__init__(
            platform_name=platform_name,
            parts_list=parts_list,
            paddle_type=paddle_type,
        )
        self._last_move = pygame.K_0
        self.paddle = paddle

    @staticmethod
    def get_validator() -> type[PaddlePlatformValidator]:
        """Returns the paddle type associated with this platform

        Returns
        -------
        PaddleType
            the paddle type associated with this platform
        """
        return PaddlePlatformValidator

    @property
    def paddle_type(self) -> PaddleType:
        """Returns the paddle type associated with this platform

        Returns
        -------
        PaddleType
            the paddle type associated with this platform
        """
        return self.config.paddle_type

    @property
    def last_move(self):
        """
        return last action in units used by pong game env
        """
        return self._last_move

    @last_move.setter
    def last_move(self, value):
        """
        set last action in units used by pong game env
        """
        self._last_move = value

    @property
    def operable(self):
        """a paddle without collision is useless so we mark it inoperable
        allows us to test inoperable code paths in commander pong
        """
        return self.paddle.collision_on
