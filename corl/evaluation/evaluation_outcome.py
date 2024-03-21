"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Contains classes to represent
"""
import dataclasses
from collections import defaultdict
from typing import Any

import pandas as pd
from pydantic import validator

from corl.evaluation.episode_artifact import EpisodeArtifact


@dataclasses.dataclass
class EvaluationOutcome:
    """Dataclass to hold the results of an evaluation run"""

    test_cases: pd.DataFrame | list[dict[str, Any]]
    episode_artifacts: dict[int, list[EpisodeArtifact]] = None  # type: ignore

    @validator("episode_artifacts", pre=True, always=True)
    def gen_default_dict(cls, v):
        if v is None:
            return defaultdict(list)
        return v

    def get_test_cases(self) -> pd.DataFrame:
        if isinstance(self.test_cases, pd.DataFrame):
            return self.test_cases
        return pd.DataFrame(self.test_cases)
