"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from collections import OrderedDict

import gymnasium

from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from corl.libraries.units import Quantity
from corl.simulators.base_simulator import BaseSimulatorState


class EpisodeLengthDoneValidator(DoneFuncBaseValidator):
    """Initialize an EpisodeLengthDone

    Parameters
    ----------
    horizon : float, optional
        The max expected length for horizon (in seconds), by default 1000
    """

    horizon: Quantity


class EpisodeLengthDone(DoneFuncBase):
    """
    CheckEpisodeLengthDone Just checks to see if we hit the timeout time and notes in the code
    via a done condition... Note this is largely a debug item
    """

    REQUIRED_UNITS = {"horizon": "second"}

    def __init__(self, **kwargs) -> None:
        self.config: EpisodeLengthDoneValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[EpisodeLengthDoneValidator]:
        """Returns the validator for this done condition"""
        return EpisodeLengthDoneValidator

    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: BaseSimulatorState,
        observation_space: gymnasium.Space,
        observation_units: OrderedDict,
    ) -> bool:
        try:
            done = next_state.sim_time >= self.config.horizon.m

        except ValueError:
            # Missing platform should trigger some other done condition
            done = False

        if done:
            next_state.episode_state[self.platform][self.name] = DoneStatusCodes.DRAW

        return done
