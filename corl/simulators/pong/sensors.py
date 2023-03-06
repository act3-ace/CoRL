"""
This module contains implementations of Sensors that reside on the Docking1dPlatform.
"""
import typing

import numpy as np
from pydantic import Field, StrictFloat, StrictStr
from typing_extensions import Annotated

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp, DiscreteProp
from corl.simulators.base_parts import BasePlatformPartValidator, BaseSensor
from corl.simulators.pong.available_platforms import PongAvailablePlatformTypes
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
    low: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [0.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=1, max_items=1)] = [1000.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["None"]
    description: str = "Position Sensor Properties"


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
    low: Annotated[typing.List[StrictFloat], Field(min_items=4, max_items=4)] = [-50.0, -50.0, -100.0, -100.0]
    high: Annotated[typing.List[StrictFloat], Field(min_items=4, max_items=4)] = [1000.0, 1000.0, 100.0, 100.0]
    unit: Annotated[typing.List[StrictStr], Field(min_items=4, max_items=4)] = ["None", "None", "None", "None"]
    description: str = "Position and Velocity Sensor Properties"


class PlatformSideProp(DiscreteProp):
    """PlatformSideProp
    """
    name: str = "side"
    n: int = 2
    unit: Annotated[typing.List[StrictStr], Field(min_items=1, max_items=1)] = ["None"]
    description: str = "Side of the platform"


class PaddlePositionSensorValidator(BasePlatformPartValidator):
    """PaddlePositionSensorValidator
    """
    paddle: PaddleType


class PaddlePositionSensor(BaseSensor):
    """
    A Sensor to measure the position of paddles.
    """

    def __init__(self, parent_platform, config, measurement_properties=PositionProp):
        self.config: PaddlePositionSensorValidator
        super().__init__(parent_platform=parent_platform, config=config, property_class=measurement_properties)

    @property
    def get_validator(self) -> typing.Type[PaddlePositionSensorValidator]:
        """
        Get the validator used to validate the kwargs passed to BaseAgent.

        Returns
        -------
        BaseAgentParser
            A BaseAgent kwargs parser and validator.
        """
        return PaddlePositionSensorValidator

    def _calculate_measurement(self, state):
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
        if self.config.paddle is PaddleType.LEFT:
            return np.array([state.pong_game.left_paddle.y], dtype=np.float32)

        return np.array([state.pong_game.right_paddle.y], dtype=np.float32)


class BallSensor(BaseSensor):
    """
    A Sensor to measure the velocity of the associated Docking1dPlatform.
    """

    def __init__(self, parent_platform, config, measurement_properties=PosVelProp):
        super().__init__(parent_platform=parent_platform, config=config, property_class=measurement_properties)

    def _calculate_measurement(self, state):
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
        return np.array(
            [state.pong_game.ball.x, state.pong_game.ball.y, state.pong_game.ball.x_vel, state.pong_game.ball.y_vel], dtype=np.float32
        )


class PlatformPaddleTypeSensor(BaseSensor):
    """
    A Sensor to the provide type of a paddle (left or right).
    """

    def __init__(self, parent_platform, config, measurement_properties=PlatformSideProp):
        super().__init__(parent_platform=parent_platform, config=config, property_class=measurement_properties)

    def _calculate_measurement(self, state):
        """get paddle side

        Parameters
        ----------
        state: BaseSimulatorState
            current simulations state

        Returns
        -------
        y: int
            current padde side
        """
        if self.parent_platform.paddle_type is PaddleType.LEFT:
            return np.array([0], dtype=np.int32)
        return np.array([1], dtype=np.int32)


# Register sensors with PluginLibrary. Requires a class, reference name, and a dict of Simulator class and platform
# type enum.

PluginLibrary.AddClassToGroup(
    PlatformPaddleTypeSensor,
    "Platform_Paddle_Type_Sensor", {
        "simulator": PongSimulator, "platform_type": PongAvailablePlatformTypes.PADDLE
    }
)

PluginLibrary.AddClassToGroup(
    PaddlePositionSensor, "Paddle_Position_Sensor", {
        "simulator": PongSimulator, "platform_type": PongAvailablePlatformTypes.PADDLE
    }
)

PluginLibrary.AddClassToGroup(BallSensor, "Ball_Sensor", {"simulator": PongSimulator, "platform_type": PongAvailablePlatformTypes.PADDLE})
