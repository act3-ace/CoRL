"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from enum import Enum

from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import TensorStructType, TensorType

from corl.policies.custom_policy import CustomPolicy, CustomPolicyValidator


class SimplePaddleControllerValidator(CustomPolicyValidator):
    """Validator for SimplePaddleController"""


class SimplePaddleController(CustomPolicy):
    """A simple pong paddle controller policy"""

    def __init__(self, observation_space, action_space, config):
        self.validated_config: SimplePaddleControllerValidator  # type: ignore
        super().__init__(observation_space, action_space, config)

    @staticmethod
    def get_validator() -> type[SimplePaddleControllerValidator]:
        """
        Get the validator for this experiment class,
        the kwargs sent to the experiment class will
        be validated using this object and add a self.config
        attr to the experiment class
        """
        return SimplePaddleControllerValidator

    def custom_compute_actions(
        self,  # noqa: PLR6301
        obs: list[TensorStructType] | TensorStructType,
        platform_obs: dict[str, dict],
        state: list[TensorType] | None = None,
        prev_action: list[TensorStructType] | TensorStructType = None,
        prev_reward: list[TensorStructType] | TensorStructType = None,
        info: dict[str, list] | None = None,
        explore: bool | None = None,
        timestep: int | None = None,
        sim_time: float | None = None,
        agent_id: str | None = None,
        epp_info: dict | None = None,
        episode: EpisodeV2 | None = None,
        **kwargs,
    ) -> TensorType:
        """Computes actions for the current policy.

        Args:
            obs: observations.
            state: List of RNN state input batches, if any.
            prev_action: previous action values.
            prev_reward: previous rewards.
            info: info objects.
            explore: Whether to pick an exploitation or exploration action.
                Set to None (default) for using the value of
                `self.config["explore"]`.
            timestep: The current (sampling) time step.

        Keyword Args:
            kwargs: Forward compatibility placeholder

        Returns:
            actions (TensorType): output actions
        """

        class Action(Enum):
            """Action space for paddle"""

            UP = 0
            NO_MOVE = 1
            DOWN = 2

        control_left_paddle = platform_obs["Obs_Platform_Paddle_Type_Sensor"]["direct_observation"] == 0
        control_right_paddle = platform_obs["Obs_Platform_Paddle_Type_Sensor"]["direct_observation"] == 1
        # ball_position_x = platform_obs['Obs_Ball_Sensor']['direct_observation'][0]
        ball_position_y = platform_obs["Obs_Ball_Sensor"]["direct_observation"][1]
        # ball_velocity_x = platform_obs['Obs_Ball_Sensor']['direct_observation'][2]
        ball_velocity_y = platform_obs["Obs_Ball_Sensor"]["direct_observation"][3]

        left_paddle_position_y = platform_obs["Obs_left_paddle"]["direct_observation"][0]
        left_paddle_height = platform_obs["Obs_left_paddle_size"]["direct_observation"][0]
        right_paddle_position_y = platform_obs["Obs_right_paddle"]["direct_observation"][0]
        right_paddle_height = platform_obs["Obs_right_paddle_size"]["direct_observation"][0]

        if control_left_paddle:
            delta_y = -(ball_position_y + ball_velocity_y - (left_paddle_position_y + left_paddle_height / 2))
        elif control_right_paddle:
            delta_y = -(ball_position_y + ball_velocity_y - (right_paddle_position_y + right_paddle_height / 2))
        else:
            raise RuntimeError("Invalid paddle")

        action: Action
        if delta_y > 0:
            action = Action.UP
        elif delta_y < 0:
            action = Action.DOWN
        else:
            action = Action.NO_MOVE

        return {"PaddleController_Paddle_move": action.value}
