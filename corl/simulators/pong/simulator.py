"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Base Simulator and Platform for Toy Openai Environments
This mainly shows a "how to use example" and provide an setup to unit test with
"""
import typing

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.units import ValueWithUnits
from corl.simulators.base_simulator import BaseSimulator, BaseSimulatorResetValidator, BaseSimulatorState
from corl.simulators.pong.paddle_platform import PaddlePlatform, PaddleType
from corl.simulators.pong.pong import GameStatus, Pong


def strip_units(item):
    """Remove units form all values with units objects
    """
    if isinstance(item, dict):
        for key, value in item.items():
            if isinstance(value, dict):
                item[key] = strip_units(value)
            elif isinstance(value, ValueWithUnits):
                item[key] = strip_units(value)
        return item

    if isinstance(item, ValueWithUnits):
        return item.value

    return item


class PongSimulatorResetValidator(BaseSimulatorResetValidator):
    """
    A Validator for Docking1dSimulatorValidator reset configs

    Parameters
    ----------
    platform_config: dict
        Contains individual initialization dicts for each agent.
        Key is platform name, value is platform's initialization dict.
    enable_health: bool
        enable or disable health, which will turn collision of the paddle off and make it inoperable
        after touching the ball
    """
    platform_config: typing.Optional[typing.Dict] = {}
    paddle0_side: PaddleType = PaddleType.LEFT
    pong: Pong = Pong()


class PongSimulatorState(BaseSimulatorState):
    """
    pong_game: the pong game being run
    game_status: the enum of the current game status
    """
    pong_game: Pong
    game_status: GameStatus = GameStatus.IN_PROGRESS


class PongSimulator(BaseSimulator):
    """
    Simulator backend for running pong game
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state: PongSimulatorState
        self._time = 0.0
        self.pong_game = None
        self._paddle0_type = None

    @property
    def get_reset_validator(self) -> typing.Type[PongSimulatorResetValidator]:
        """Return validator"""
        return PongSimulatorResetValidator

    def reset(self, config):
        config = self.get_reset_validator(**strip_units(config))

        self._time = 0.0

        self.pong_game = config.pong

        self._paddle0_type = PaddleType(config.paddle0_side)

        paddle_dict = self._initialize_paddles(config.platforms)

        sim_platforms = {}  # pylint: disable=attribute-defined-outside-init
        for agent_id, agent_config in self.config.agent_configs.items():
            agent_config = self.config.agent_configs[agent_id]
            sim_platforms[agent_id] = PaddlePlatform(
                platform_name=agent_id,
                parts_list=agent_config.parts_list,
                paddle_type=self._paddle_type(agent_id),
                paddle=paddle_dict[agent_id]
            )

        self._state = PongSimulatorState(sim_platforms=sim_platforms, sim_time=self._time, pong_game=self.pong_game)
        self.update_sensor_measurements()
        return self._state

    def step(self):
        keys = []
        for platform in self.platforms.values():
            keys.append(platform.last_move)

        self._state.game_status = self.pong_game.step(keys)
        self._time += 1
        self._state.pong_game = self.pong_game
        self._state.sim_time = self._time
        self.update_sensor_measurements()
        return self._state

    def update_sensor_measurements(self):
        """
        Update and cache all the measurements of all the sensors on each platform
        """
        for plat in self._state.sim_platforms.values():
            for sensor in plat.sensors.values():
                sensor.calculate_and_cache_measurement(state=self._state)

    @property
    def sim_time(self) -> float:
        return self._time

    @property
    def platforms(self):
        return self._state.sim_platforms

    def mark_episode_done(self, done_info, episode_state):
        pass

    def save_episode_information(self, dones, rewards, observations):
        pass

    def render(self, state, mode="human"):  # pylint: disable=unused-argument
        """only render first environment
        """
        return

    def _paddle_type(self, platform_id):
        if platform_id == "paddle0":
            return self._paddle0_type

        if platform_id == "paddle1":
            if self._paddle0_type == PaddleType.LEFT:
                return PaddleType.RIGHT
            if self._paddle0_type == PaddleType.RIGHT:
                return PaddleType.LEFT

        raise RuntimeError(f"Invalid paddle types for {platform_id}")

    def _get_agent_platform(self, platform_id):
        return self._pong_paddle(strip_units(self._paddle_type(platform_id)))

    def _initialize_paddles(self, platform_configs):
        if self._paddle0_type is PaddleType.LEFT:
            paddle0 = self.pong_game.left_paddle
            paddle1 = self.pong_game.right_paddle
        else:
            paddle0 = self.pong_game.right_paddle
            paddle1 = self.pong_game.left_paddle

        for platform, platform_config in platform_configs.items():
            if platform == "paddle0":
                paddle0 = paddle0.copy(update=strip_units(platform_config))
            if platform == "paddle1":
                paddle1 = paddle1.copy(update=strip_units(platform_config))

        if self._paddle0_type is PaddleType.LEFT:
            self.pong_game.left_paddle = paddle0
            self.pong_game.right_paddle = paddle1
        else:
            self.pong_game.left_paddle = paddle1
            self.pong_game.right_paddle = paddle0

        return {"paddle0": paddle0, "paddle1": paddle1}


PluginLibrary.AddClassToGroup(PongSimulator, "PongSimulator", {})
