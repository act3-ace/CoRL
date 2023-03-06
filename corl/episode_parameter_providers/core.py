"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import abc
import os
import typing

from numpy.random import Generator, RandomState
from pydantic import BaseModel

from corl.libraries.parameters import Parameter

PathLike = typing.Union[str, os.PathLike]
Randomness = typing.Union[Generator, RandomState]
ParameterModel = typing.Mapping[typing.Tuple[str, ...], Parameter]


class EpisodeParameterProviderValidator(BaseModel):
    """Validation model for the inputs of EpisodeParameterProvider"""
    parameters: ParameterModel = {}

    class Config:
        """Allow arbitrary types for Parameter"""
        arbitrary_types_allowed = True
        allow_mutation = False


class EpisodeParameterProvider(abc.ABC):
    """Interface definition for episode parameter providers.
    """

    def __init__(self, **kwargs) -> None:
        self.config: EpisodeParameterProviderValidator = self.get_validator(**kwargs)

    @property
    def get_validator(self) -> typing.Type[EpisodeParameterProviderValidator]:
        """Get the validator for this class."""
        return EpisodeParameterProviderValidator

    def get_params(self, rng: Randomness) -> typing.Tuple[ParameterModel, typing.Union[int, None]]:
        """Get the next instance of episode parameters from this provider

        Subclasses: Override _do_get_params.

        Parameters
        ----------
        rng : Union[Generator, RandomState]
            Random number generator from which to draw random values.

        Returns
        -------
        ParameterModel
            The parameters for this episode
        episode_id
            The episode index number for this set of parameters
        """

        output, episode_id = self._do_get_params(rng)

        extra_keys = output.keys() - self.config.parameters.keys()
        if extra_keys:
            raise KeyError(f'Extra keys provided: {extra_keys}')

        missing_keys = self.config.parameters.keys() - output.keys()
        if missing_keys:
            raise KeyError(f'Missing keys: {missing_keys}')

        bad_types: typing.Dict[str, str] = {}
        for key, value in output.items():
            if not isinstance(value, Parameter):
                bad_types['.'.join(key)] = type(value).__name__
        if bad_types:
            raise TypeError(f'Unsupported types: {bad_types}')

        return output, episode_id

    @abc.abstractmethod
    def _do_get_params(self, rng: Randomness) -> typing.Tuple[ParameterModel, typing.Union[int, None]]:
        """Get the next instance of episode parameters from this provider

        This is an abstract method that must be overridden by each subclass.

        DO NOT CALL DIRECTLY.  USE get_params.

        Parameters
        ----------
        rng : Union[Generator, RandomState]
            Random number generator from which to draw random values.

        Returns
        -------
        ParameterCollection
            The parameters for this episode
        episode_id
            The episode index number for this set of parameters
        """
        raise NotImplementedError

    def compute_metrics(self) -> typing.Dict[str, typing.Any]:
        """Get metrics on the operation of this provider.

        Often used in `on_episode_end` training callbacks.
        """
        return {}

    def update(self, results: dict, rng: Randomness) -> None:
        """Update the operation of this provider.

        Often used in `on_train_result` training callbacks.

        Parameters
        ----------
        results : dict
            As described by ray.rllib.algorithms.callbacks.DefaultCallbacks.on_train_result.
            See https://docs.ray.io/en/master/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks.on_train_result
        rng : Union[Generator, RandomState]
            Random number generator from which to draw random values.
        """

    def save_checkpoint(self, checkpoint_path: PathLike) -> None:
        """Save the internal state of the parameter provider.

        Parameters
        ----------
        checkpoint_path : PathLike
            Filesystem path at which to save the checkpoint
        """

    def load_checkpoint(self, checkpoint_path: PathLike) -> None:
        """Load the internal state from a checkpoint.

        Parameters
        ----------
        checkpoint_path : PathLike
            Filesystem path from which to restore the checkpoint
        """
