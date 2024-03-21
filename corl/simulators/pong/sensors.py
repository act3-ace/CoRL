"""
This module contains implementations of Sensors that reside on the Docking1dPlatform.
"""
from typing import Annotated

import numpy as np
from pydantic import Field, StrictFloat

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp, DiscreteProp
from corl.libraries.units import Quantity, corl_get_ureg
from corl.simulators.base_parts import BasePlatformPartValidator, BaseSensor
from corl.simulators.pong.available_platforms import PongAvailablePlatformType
from corl.simulators.pong.paddle_platform import PaddleType
from corl.simulators.pong.simulator import PongSimulator


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
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [1000.0]
    unit: str = "dimensionless"
    description: str = "Position Sensor Properties"


class SizeProp(BoxProp):
    """
    paddle size sensor properties.

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

    name: str = "paddle_size"
    low: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [0.0]
    high: Annotated[list[StrictFloat], Field(min_length=1, max_length=1)] = [1000.0]
    unit: str = "dimensionless"
    description: str = "paddle Size Sensor Properties"


class PosVelProp(BoxProp):
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

    name: str = "ball_state"
    low: Annotated[list[StrictFloat], Field(min_length=4, max_length=4)] = [-50.0, -50.0, -100.0, -100.0]
    high: Annotated[list[StrictFloat], Field(min_length=4, max_length=4)] = [1000.0, 1000.0, 100.0, 100.0]
    unit: str = "dimensionless"
    description: str = "Position and Velocity Sensor Properties"


class PlatformSideProp(DiscreteProp):
    """PlatformSideProp"""

    name: str = "side"
    n: int = 2
    unit: str = "dimensionless"
    description: str = "Side of the platform"


class PaddlePositionSensorValidator(BasePlatformPartValidator):
    """PaddlePositionSensorValidator"""

    paddle: PaddleType


class PaddlePositionSensor(BaseSensor):
    """
    A Sensor to measure the position of paddles.
    """

    def __init__(self, parent_platform, config, measurement_properties=PositionProp) -> None:
        self.config: PaddlePositionSensorValidator
        super().__init__(parent_platform=parent_platform, config=config, property_class=measurement_properties)

    @staticmethod
    def get_validator() -> type[PaddlePositionSensorValidator]:
        """
        Get the validator used to validate the kwargs passed to BaseAgent.

        Returns
        -------
        BaseAgentParser
            A BaseAgent kwargs parser and validator.
        """
        return PaddlePositionSensorValidator

    def _calculate_measurement(self, state) -> Quantity:
        """get paddle y position

        Parameters
        ----------
        state: BaseSimulatorState
            current simulations state

        Returns
        -------
        y: np.array
            current y position of paddle
        """
        val = state.pong_game.right_paddle.y
        if self.config.paddle is PaddleType.LEFT:
            val = state.pong_game.left_paddle.y
        return corl_get_ureg().Quantity(np.array([val], dtype=np.float32), self._properties.unit)


class PaddleSizeSensorValidator(BasePlatformPartValidator):
    """PaddleSizeSensorValidator"""

    paddle: PaddleType


class PaddleSizeSensor(BaseSensor):
    """
    A Sensor to measure the size of paddles.
    """

    def __init__(self, parent_platform, config, measurement_properties=SizeProp):
        self.config: PaddleSizeSensorValidator  # type: ignore
        super().__init__(parent_platform=parent_platform, config=config, property_class=measurement_properties)

    @staticmethod
    def get_validator() -> type[PaddleSizeSensorValidator]:
        """
        Get the validator used to validate the kwargs passed to BaseAgent.

        Returns
        -------
        BaseAgentParser
            A BaseAgent kwargs parser and validator.
        """
        return PaddleSizeSensorValidator

    def _calculate_measurement(self, state) -> Quantity:
        """get paddle y size

        Parameters
        ----------
        state: BaseSimulatorState
            current simulations state

        Returns
        -------
        y: np.array
            y size of paddle
        """
        val = state.pong_game.right_paddle.height
        if self.config.paddle is PaddleType.LEFT:
            val = state.pong_game.left_paddle.height
        return corl_get_ureg().Quantity(np.array([val], dtype=np.float32), self._properties.unit)


class BallSensor(BaseSensor):
    """
    A Sensor to measure the velocity of the associated Docking1dPlatform.
    """

    def __init__(self, parent_platform, config, measurement_properties=PosVelProp):
        super().__init__(parent_platform=parent_platform, config=config, property_class=measurement_properties)

    def _calculate_measurement(self, state) -> Quantity:  # noqa: PLR6301
        """get paddle y position

        Parameters
        ----------
        state: BaseSimulatorState
            current simulations state

        Returns
        -------
        ball_state: np.array
            four element array contain x, y position and velocity of the ball
        """
        return corl_get_ureg().Quantity(
            np.array(
                [state.pong_game.ball.x, state.pong_game.ball.y, state.pong_game.ball.x_vel, state.pong_game.ball.y_vel], dtype=np.float32
            ),
            "dimensionless",
        )


class PlatformPaddleTypeSensor(BaseSensor):
    """
    A Sensor to the provide type of a paddle (left or right).
    """

    def __init__(self, parent_platform, config, measurement_properties=PlatformSideProp):
        super().__init__(parent_platform=parent_platform, config=config, property_class=measurement_properties)

    def _calculate_measurement(self, state) -> Quantity:
        """get paddle side

        Parameters
        ----------
        state: BaseSimulatorState
            current simulations state

        Returns
        -------
        y: int
            current pade side
        """
        val = 1
        if self.parent_platform.paddle_type is PaddleType.LEFT:
            val = 0
        return corl_get_ureg().Quantity(np.array(val, dtype=np.int32), self._properties.unit)


# Register sensors with PluginLibrary. Requires a class, reference name, and a dict of Simulator class and platform
# type enum.

PluginLibrary.AddClassToGroup(
    PlatformPaddleTypeSensor, "Platform_Paddle_Type_Sensor", {"simulator": PongSimulator, "platform_type": PongAvailablePlatformType}
)

PluginLibrary.AddClassToGroup(
    PaddlePositionSensor, "Paddle_Position_Sensor", {"simulator": PongSimulator, "platform_type": PongAvailablePlatformType}
)

PluginLibrary.AddClassToGroup(
    PaddleSizeSensor, "Paddle_Size_Sensor", {"simulator": PongSimulator, "platform_type": PongAvailablePlatformType}
)

PluginLibrary.AddClassToGroup(BallSensor, "Ball_Sensor", {"simulator": PongSimulator, "platform_type": PongAvailablePlatformType})
