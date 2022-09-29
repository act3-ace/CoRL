"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import typing
from collections import OrderedDict

from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from corl.libraries.environment_dict import DoneDict
from corl.libraries.state_dict import StateDict
from corl.libraries.units import Time


class EpisodeLengthDoneValidator(DoneFuncBaseValidator):
    """Initialize an EpisodeLengthDone

    Parameters
    ----------
    horizon : float, optional
        The max expected length for horizon (in seconds), by default 1000
    """
    horizon: float = 1000


class EpisodeLengthDone(DoneFuncBase):
    """
    CheckEpisodeLengthDone Just checks to see if we hit the timeout time and notes in the code
    via a done condition... Note this is largely a debug item
    """

    REQUIRED_UNITS = {'horizon': Time.Second}

    def __init__(self, **kwargs) -> None:
        self.config: EpisodeLengthDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[EpisodeLengthDoneValidator]:
        """Returns the validator for this done condition"""
        return EpisodeLengthDoneValidator

    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> DoneDict:

        done = DoneDict()
        try:
            done[self.platform] = next_state['sim_time'] >= self.config.horizon

            if done[self.platform]:
                next_state.episode_state[self.platform][self.name] = DoneStatusCodes.DRAW

        except ValueError:
            # Missing platform should trigger some other done condition
            done[self.platform] = False

        self._set_all_done(done)

        if done[self.platform]:
            next_state.episode_state[self.platform][self.name] = DoneStatusCodes.DRAW

        return done
