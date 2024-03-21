"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import copy
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import validator

from corl.episode_parameter_providers import EpisodeParameterProvider, EpisodeParameterProviderValidator, ParameterModel, Randomness
from corl.evaluation.connection.signal import BaseSlot, Signal
from corl.libraries.parameters import OverridableParameterWrapper

T = TypeVar("T")


@dataclass
class EppUpdate(Generic[T]):
    test_case_index: int
    update: T


class EppUpdateSignaller(Generic[T]):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.epp_update_signal: Signal[EppUpdate[T]] = Signal[EppUpdate[T]]()  # type: ignore


class NetworkedParameterProviderValidator(EpisodeParameterProviderValidator, Generic[T]):
    class Config:
        arbitrary_types_allowed = True

    registrar: EppUpdateSignaller[T]
    epp_variety: Literal["environment", "agent"]
    agent_id: str | None = None

    @validator("agent_id")
    def validate_agent_id(cls, v, values):
        if values["epp_variety"] == "agent" and not v:
            raise ValueError("agent_id must not be None or '' if the epp_variety is 'agent'")
        return v


class BaseNetworkParameterProvider(EpisodeParameterProvider, BaseSlot[EppUpdate[T]]):
    """EpisodeParameterProvider that receives values remotely"""

    def __init__(self, **kwargs):
        self.config: NetworkedParameterProviderValidator[T]  # type: ignore
        super().__init__(**kwargs)
        print(f"BaseNetworkParameterProvider::__init__ {id(self)}")

        self.base_params = {k: OverridableParameterWrapper(v) for k, v in self.config.parameters.items()}
        self.cur_params = copy.deepcopy(self.base_params)
        self._test_case_idx = 0

        self.config.registrar.epp_update_signal.register(self)

    @staticmethod
    def get_validator() -> type[EpisodeParameterProviderValidator]:
        return NetworkedParameterProviderValidator[T]

    def reset(self):
        self.cur_params = copy.deepcopy(self.base_params)
        self._test_case_idx = 0

    def shutdown(self):
        # del self.config.registrar
        ...

    def _do_get_params(self, rng: Randomness, env_epp_ctx: dict | None) -> tuple[ParameterModel, int | None, dict | None]:
        print(f"BaseNetworkParameterProvider::_do_get_params {id(self)} {self._test_case_idx}")
        return self.cur_params, self._test_case_idx, env_epp_ctx

    def on_message(self, epp_update: EppUpdate[T]):
        print(f"BaseNetworkParameterProvider::on_message {id(self)} {epp_update.test_case_index}")
        self.cur_params = self._do_update_params(copy.deepcopy(self.base_params), epp_update.update)
        self._test_case_idx = epp_update.test_case_index

    @abstractmethod
    def _do_update_params(
        self, params: Mapping[tuple[str, ...], OverridableParameterWrapper], updates: T
    ) -> Mapping[tuple[str, ...], OverridableParameterWrapper]:
        ...

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     self.config.registrar.epp_update_signal.register(Slot(self))


class TestNetworkedParameterProvider(BaseNetworkParameterProvider):
    def _do_update_params(
        self, params: Mapping[tuple[str, ...], OverridableParameterWrapper], updates  # noqa: PLR6301
    ) -> Mapping[tuple[str, ...], OverridableParameterWrapper]:
        return params
