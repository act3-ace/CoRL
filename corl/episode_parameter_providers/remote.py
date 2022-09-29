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
from functools import lru_cache

import numpy as np
import ray
from gym.utils import seeding
from numpy.random import Generator, RandomState
from pydantic import validator

from corl.episode_parameter_providers import EpisodeParameterProvider, EpisodeParameterProviderValidator, ParameterModel, Randomness
from corl.libraries.factory import Factory


class RayActorProxy:
    """Creates a 'proxy' to a named ray actor"""

    def __init__(self, actor_id: str, namespace: typing.Optional[str] = None):
        self._actor_id = actor_id
        self._namespace = namespace

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def actor(self) -> ray.actor.ActorHandle:
        """Retrieves the named ray actor"""
        if not ray.is_initialized():
            raise RuntimeError('Cannot get named actor without ray')
        return ray.get_actor(self._actor_id, self._namespace)


class RemoteEpisodeParameterProviderValidator(EpisodeParameterProviderValidator):
    """Validation model for the inputs of RemoteEpisodeParameterProvider"""
    internal_class: typing.Type[EpisodeParameterProvider]
    internal_config: typing.Dict[str, typing.Any] = {}
    actor_name: str
    namespace: typing.Optional[str] = None

    @validator('internal_class')
    def internal_not_remote(cls, v):
        """Confirm that internal class is not also remote"""
        assert v != RemoteEpisodeParameterProvider
        return v


class RemoteEpisodeParameterProvider(EpisodeParameterProvider):
    """Wrap EpisodeParameterProvider as a ray remote actor and manage data passing between ray processes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = typing.cast(RemoteEpisodeParameterProviderValidator, self.config)

        ray.remote(self.config.internal_class
                   ).options(  # type: ignore
                       name=self.config.actor_name,
                       namespace=self.config.namespace,
                       lifetime='detached',
                   ).remote(**self.config.internal_config, **kwargs)  # noqa: E126 I could not get this to format

        self._rap = RayActorProxy(self.config.actor_name, namespace=self.config.namespace)

    @property
    def get_validator(self) -> typing.Type[RemoteEpisodeParameterProviderValidator]:
        return RemoteEpisodeParameterProviderValidator

    @staticmethod
    def wrap_epp_factory(epp_factory: Factory, actor_name: str, namespace: str = None) -> Factory:
        """Wraps an existing EpisodeParameterProvider Factory as a RemoteEpisodeParameterProvider"""
        if not issubclass(epp_factory.type, EpisodeParameterProvider):  # type: ignore
            raise TypeError(f"Invalid Factory.type: {epp_factory.type}, {EpisodeParameterProvider.__qualname__} required")

        remote_config = {
            'namespace': namespace, 'actor_name': actor_name, 'internal_class': epp_factory.type, 'internal_config': epp_factory.config
        }

        return Factory(type=RemoteEpisodeParameterProvider, config=remote_config)

    def kill_actor(self) -> None:
        """Kill the underlying actor used by this provider."""
        try:
            ray.kill(self._rap.actor)
        except ray.exceptions.RaySystemError:
            pass

    def _do_get_params(self, rng: Randomness) -> typing.Tuple[ParameterModel, typing.Union[int, None]]:

        if isinstance(rng, Generator):
            seed = rng.integers(low=0, high=1000000)
            new_rng = np.random.default_rng(seed=seed)
        elif isinstance(rng, RandomState):
            seed = rng.randint(low=0, high=1000000)
            new_rng, _ = seeding.np_random(seed)
        else:
            raise RuntimeError(f"rng type provided to function was {rng}, but this class only knows numpy Generator or RandomState")

        return ray.get(self._rap.actor.get_params.remote(new_rng))  # type: ignore

    def compute_metrics(self) -> typing.Dict[str, typing.Any]:
        return ray.get(self._rap.actor.compute_metrics.remote())  # type: ignore

    def update(self, results: dict, rng: Randomness) -> None:

        if isinstance(rng, Generator):
            seed = rng.integers(low=0, high=1000000)
            new_rng = np.random.default_rng(seed=seed)
        elif isinstance(rng, RandomState):
            seed = rng.randint(low=0, high=1000000)
            new_rng, _ = seeding.np_random(seed)
        else:
            raise RuntimeError(f"rng type provided to function was {rng}, but this class only knows numpy Generator or RandomState")

        ray.get(self._rap.actor.update.remote(results, new_rng))  # type: ignore

    def save_checkpoint(self, checkpoint_path) -> None:
        ray.get(self._rap.actor.save_checkpoint.remote(checkpoint_path))  # type: ignore

    def load_checkpoint(self, checkpoint_path) -> None:
        ray.get(self._rap.actor.load_checkpoint.remote(checkpoint_path))  # type: ignore
