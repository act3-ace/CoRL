"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import logging
from abc import abstractmethod
from typing import Annotated, Generic, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict

from corl.evaluation.connection.signal import Signal
from corl.evaluation.launchers.base_eval import EvalConfig
from corl.libraries.factory import Factory

EppUpdateT = TypeVar("EppUpdateT")


class BaseEvalConnectionValidator(BaseModel):
    ...


class BaseEvalConnection(Generic[EppUpdateT]):
    def __init__(self, **kwargs):
        self._config = self.get_validator()(**kwargs)
        self._logger = logging.getLogger(type(self).__name__)

        self.reset_signal: Signal[EvalConfig] = Signal[EvalConfig]()  # type: ignore
        self.modify_epp_signal: Signal[EppUpdateT] = Signal[EppUpdateT]()  # type: ignore

    @staticmethod
    def get_validator() -> type[BaseEvalConnectionValidator]:
        return BaseEvalConnectionValidator

    @abstractmethod
    def stop(self):
        """Stops the connection"""


class NullConnection(BaseEvalConnection):
    def stop(self):
        pass


class ConnectionValidator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: Annotated[BaseEvalConnection, BeforeValidator(Factory.resolve_factory)] = NullConnection()
