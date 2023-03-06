"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Done condition for OpenAIGymSimulator
"""
from corl.dones.done_func_base import DoneFuncBase
from corl.libraries.environment_dict import DoneDict


class OpenAIGymDone(DoneFuncBase):
    """
    A done functor that simply mirrors the done condition coming from the sim state
    this only works with the OpenAIGymSimulator
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
        done = DoneDict()
        done[self.config.agent_name] = next_state.dones[self.config.platform_name]
        self._set_all_done(done)
        return done
