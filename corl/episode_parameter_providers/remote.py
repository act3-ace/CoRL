"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import contextlib
import functools
import typing

import numpy as np
import ray
from gymnasium.utils import seeding
from numpy.random import Generator, RandomState
from pydantic import field_validator

from corl.episode_parameter_providers import EpisodeParameterProvider, EpisodeParameterProviderValidator, ParameterModel, Randomness
from corl.libraries.factory import Factory


class RayActorProxy:
    """Creates a 'proxy' to a named ray actor"""

    def __init__(self, actor_id: str, namespace: str | None = None):
        self._actor_id = actor_id
        self._namespace = namespace

    @functools.cached_property
    def actor(self) -> ray.actor.ActorHandle:
        """Retrieves the named ray actor"""
        if not ray.is_initialized():
            raise RuntimeError("Cannot get named actor without ray")
        return ray.get_actor(self._actor_id, self._namespace)


class RemoteEpisodeParameterProviderValidator(EpisodeParameterProviderValidator):
    """Validation model for the inputs of RemoteEpisodeParameterProvider"""

    internal_class: type[EpisodeParameterProvider]
    internal_config: dict[str, typing.Any] = {}
    actor_name: str
    namespace: str | None = None

    @field_validator("internal_class")
    @classmethod
    def internal_not_remote(cls, v):
        """Confirm that internal class is not also remote"""
        assert v != RemoteEpisodeParameterProvider
        return v


class RemoteEpisodeParameterProvider(EpisodeParameterProvider):
    """Wrap EpisodeParameterProvider as a ray remote actor and manage data passing between ray processes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = typing.cast(RemoteEpisodeParameterProviderValidator, self.config)

        ray.remote(self.config.internal_class).options(  # type: ignore
            name=self.config.actor_name,
            namespace=self.config.namespace,
            lifetime="detached",
        ).remote(
            **self.config.internal_config, **kwargs
        )  # I could not get this to format

        self._rap = RayActorProxy(self.config.actor_name, namespace=self.config.namespace)

    @staticmethod
    def get_validator() -> type[RemoteEpisodeParameterProviderValidator]:
        return RemoteEpisodeParameterProviderValidator

    @staticmethod
    def wrap_epp_factory(epp_factory: Factory, actor_name: str, namespace: str | None = None) -> Factory:
        """Wraps an existing EpisodeParameterProvider Factory as a RemoteEpisodeParameterProvider"""
        if not issubclass(epp_factory.type, EpisodeParameterProvider):
            raise TypeError(f"Invalid Factory.type: {epp_factory.type}, {EpisodeParameterProvider.__qualname__} required")

        remote_config = {
            "namespace": namespace,
            "actor_name": actor_name,
            "internal_class": epp_factory.type,
            "internal_config": epp_factory.config,
        }

        return Factory(type=RemoteEpisodeParameterProvider, config=remote_config)

    def kill_actor(self) -> None:
        """Kill the underlying actor used by this provider."""
        with contextlib.suppress(ray.exceptions.RaySystemError):
            ray.kill(self._rap.actor)

    def reset(self):
        ray.get(self._rap.actor.reset.remote())

    def shutdown(self):
        # self.kill_actor()
        ...

    def _do_get_params(self, rng: Randomness, env_epp_ctx: dict | None) -> tuple[ParameterModel, int | None, dict | None]:
        if isinstance(rng, Generator):
            seed = rng.integers(low=0, high=1000000)
            new_rng = np.random.default_rng(seed=seed)
        elif isinstance(rng, RandomState):
            seed = rng.randint(low=0, high=1000000)
            new_rng, _ = seeding.np_random(seed)
        else:
            raise RuntimeError(f"rng type provided to function was {rng}, but this class only knows numpy Generator or RandomState")

        return ray.get(self._rap.actor.get_params.remote(new_rng, env_epp_ctx))

    def compute_metrics(self) -> dict[str, typing.Any]:
        return ray.get(self._rap.actor.compute_metrics.remote())

    def update(self, results: dict, rng: Randomness) -> None:
        if isinstance(rng, Generator):
            seed = rng.integers(low=0, high=1000000)
            new_rng = np.random.default_rng(seed=seed)
        elif isinstance(rng, RandomState):
            seed = rng.randint(low=0, high=1000000)
            new_rng, _ = seeding.np_random(seed)
        else:
            raise RuntimeError(f"rng type provided to function was {rng}, but this class only knows numpy Generator or RandomState")

        ray.get(self._rap.actor.update.remote(results, new_rng))

    def save_checkpoint(self, checkpoint_path) -> None:
        ray.get(self._rap.actor.save_checkpoint.remote(checkpoint_path))

    def load_checkpoint(self, checkpoint_path) -> None:
        ray.get(self._rap.actor.load_checkpoint.remote(checkpoint_path))
