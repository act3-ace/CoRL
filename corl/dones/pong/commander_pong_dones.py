"""
This module defines functions that determine terminal conditions for the 1D Docking environment.
"""

from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from corl.rewards.base_measurement_operation import ExtractorSet, ObservationExtractorValidator
from corl.simulators.common_platform_utils import get_platform_by_name
from corl.simulators.pong.pong import GameStatus


class CommanderPongGameDoneFunctionValidator(DoneFuncBaseValidator):
    """
    observation: observation extractor path to make sure a platform is still operable
    """

    observation: ObservationExtractorValidator


class CommanderPongGameDoneFunction(DoneFuncBase):
    """
    A done function that determines if the game status is no longer in progress and
    determines if the paddle this done is assigned to won or lost
    """

    def __init__(self, **kwargs) -> None:
        self.config: CommanderPongGameDoneFunctionValidator
        super().__init__(**kwargs)
        self.extractor: ExtractorSet = self.config.observation.construct_extractors()

    @staticmethod
    def get_validator() -> type[CommanderPongGameDoneFunctionValidator]:
        return CommanderPongGameDoneFunctionValidator

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
        try:
            obs = self.extractor.value(observation).m.item()
        except:  # noqa: E722
            return False

        game_status = next_state.game_status

        done = False
        if game_status is not GameStatus.IN_PROGRESS:
            done = True
            if (game_status == GameStatus.LEFT_WIN and obs == 1) or (game_status == GameStatus.RIGHT_WIN and obs == 0):
                next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.WIN
            else:
                next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.LOSE

        return done


class CommanderPongHealthDone(DoneFuncBase):
    """
    A done function that determines if the current paddle has no health
    and needs to be set to done
    """

    def __call__(self, observation, action, next_observation, next_state, observation_space, observation_units):
        # Find Target Platform
        platform = get_platform_by_name(next_state, self.platform, allow_invalid=True)

        # platform does not exist
        if platform is None:
            return False

        return platform.paddle.current_health <= 0
