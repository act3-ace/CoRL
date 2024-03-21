"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
This module defines an evaluation specific ParameterProvider, which wraps a given ParameterProvider
and delegates parameter value generation to it. It increments an internal 'index' attribute up to
the provided 'num_test_cases' int, to provide an accurate episode_id during evaluation.

Author: John McCarroll
"""


from pydantic import ImportString

from corl.episode_parameter_providers import EpisodeParameterProvider, EpisodeParameterProviderValidator, ParameterModel, Randomness


class IncrementalWrapperValidator(EpisodeParameterProviderValidator):
    """
    Validation model for the inputs of IncrementalParameterProviderWrapper
    """

    num_test_cases: int
    type: ImportString  # noqa: A003 # the class of the wrapped ParameterProvider
    config: dict | None = {}  # the config for the wrapped ParameterProvider


class IncrementalParameterProviderWrapper(EpisodeParameterProvider):
    """
    A simple ParameterProvider wrapper, which delegates parameter value generation to the ParameterProvider it maintains.
    The added feature is an incremented 'index' attribute, used to track episode_id for the Evaluation framework.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.index = 0
        config = self.config.config if self.config.config is not None else {}  # type: ignore
        self.epp = self.config.type(type=self.config.type, parameters=self.config.parameters, **config)  # type: ignore

    def reset(self) -> None:
        self.index = 0

    @staticmethod
    def get_validator() -> type[IncrementalWrapperValidator]:
        """Get the validator for this class."""
        return IncrementalWrapperValidator

    def _do_get_params(self, rng: Randomness, env_epp_ctx: dict | None) -> tuple[ParameterModel, int | None, dict | None]:
        """
        This method retrieves parameter values from its wrapped ParameterProvider.
        It then returns them, alongside the episode_id.
        """

        # delegate to get params
        params, _, env_epp_ctx = self.epp.get_params(rng, env_epp_ctx)

        # get episode_id
        episode_id = self.index if self.index < self.config.num_test_cases else None  # type: ignore

        # increment after params collected
        self.index += 1

        return params, episode_id, env_epp_ctx

    # TODO: delegate other methods to wrapped EPP?
