"""
This module defines functions that determine terminal conditions for the 1D Docking environment.
"""

from corl.dones.done_func_base import DoneFuncBase, DoneStatusCodes
from corl.simulators.common_platform_utils import get_platform_by_name
from corl.simulators.pong.paddle_platform import PaddleType
from corl.simulators.pong.pong import GameStatus


class PongGameDoneFunction(DoneFuncBase):
    """
    A done function that determines if the game status is no longer in progress and
    determines if the paddle this done is assigned to won or lost
    """

    def __call__(self, observation, action, next_observation, next_state, observation_space, observation_units):
        """
        Parameters
        ----------
        observation : np.ndarray
            np.ndarray describing the current observation
        action : np.ndarray
            np.ndarray describing the current action
        next_observation : np.ndarray
            np.ndarray describing the incoming observation
        next_state : np.ndarray
            np.ndarray describing the incoming state

        Returns
        -------
        done : DoneDict
            dictionary containing the condition condition for the current agent

        """

        paddle_type = get_platform_by_name(next_state, self.config.platform_name).paddle_type
        game_status = next_state.game_status

        done = False
        if game_status is not GameStatus.IN_PROGRESS:
            done = True
            if self._win_left(game_status, paddle_type) or self._win_right(game_status, paddle_type):
                next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.WIN
            else:
                next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.LOSE
        # else:
        #     next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.DRAW

        return done

    @staticmethod
    def _win_left(game_status: GameStatus, paddle_type: PaddleType):
        """Check if the paddle_type is left and left one the game

        Parameters
        ----------
        game_status : GameStatus
            status of current game
        paddle_type : PaddleType
            paddle type of the platform being check

        Returns
        -------
        done : bool
            return true if the paddle type one the game
        """

        return game_status == GameStatus.LEFT_WIN and paddle_type is PaddleType.LEFT

    @staticmethod
    def _win_right(game_status: GameStatus, paddle_type: PaddleType):
        """Check if the paddle_type is right and right one the game

        Parameters
        ----------
        game_status : GameStatus
            status of current game
        paddle_type : PaddleType
            paddle type of the platform being check

        Returns
        -------
        done : bool
            return true if the paddle type one the game
        """
        return game_status == GameStatus.RIGHT_WIN and paddle_type is PaddleType.RIGHT
