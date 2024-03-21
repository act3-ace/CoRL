"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from abc import abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

from pydantic import BaseModel
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks

T = TypeVar("T")


class InferenceCallbackValidator(BaseModel):
    """Base validator for InferenceCallback"""


class InferenceCallback(DefaultCallbacks, Generic[T]):
    def __init__(self, legacy_callbacks_dict: dict[str, Callable] | None = None):
        super().__init__(legacy_callbacks_dict)  # type:  ignore[arg-type]

    @staticmethod
    def get_validator() -> type[InferenceCallbackValidator]:
        """Return validator"""
        return InferenceCallbackValidator

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        super().on_algorithm_init(algorithm=algorithm, kwargs=kwargs)

        self.config = self.get_validator()(**dict(algorithm.config).get(f"{type(self).__name__}", {}))

    @abstractmethod
    def process_message(self, message: T):
        ...
