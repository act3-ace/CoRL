"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Done condition for GymnasiumSimulator
"""
from corl.dones.done_func_base import DoneFuncBase


class GymnasiumDone(DoneFuncBase):
    """
    A done functor that simply mirrors the done condition coming from the sim state
    this only works with the GymnasiumSimulator
    """

    def __call__(
        self,
        observation,
        action,
        next_observation,
        next_state,
        observation_space,
        observation_units,
    ):
        return next_state.terminated[self.config.platform_name] or next_state.truncated[self.config.platform_name]
