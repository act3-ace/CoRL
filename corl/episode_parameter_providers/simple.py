"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""


from corl.episode_parameter_providers import EpisodeParameterProvider, ParameterModel, Randomness


class SimpleParameterProvider(EpisodeParameterProvider):
    """EpisodeParameterProvider that does nothing but return the default."""

    def reset(self):
        ...

    def _do_get_params(self, rng: Randomness, env_epp_ctx: dict | None) -> tuple[ParameterModel, int | None, dict | None]:
        return self.config.parameters, None, env_epp_ctx
