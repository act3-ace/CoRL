"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Contains abstract interfaces defining how to record an Evaluation outcome
"""
import typing
from abc import abstractmethod
from pathlib import Path

from corl.evaluation.evaluation_outcome import EvaluationOutcome


class IRecord:
    """Interface that represents a record that can save and load EvaluationOutcomes"""

    @property
    @abstractmethod
    def absolute_path(self) -> Path:
        """Get the absolute path related to this record"""

    @abstractmethod
    def save(self, outcome: EvaluationOutcome) -> None:
        """Save evaluation outcome

        Arguments:
            outcome {EvaluationOutcome} -- Evaluation outcome to save
        """

    @abstractmethod
    def load(self) -> EvaluationOutcome:
        """Load EvaluationOutcome

        Returns:
            EvaluationOutcome -- Evaluation Outcome loaded from folder
        """


T = typing.TypeVar("T", bound=IRecord)


class IRecorder(typing.Generic[T]):
    """Interface the represents a recorder

    A recorder generates a Record
    """

    @abstractmethod
    def resolve(self) -> T:
        """Resolve the recorder and generate a Record"""
