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
import typing

import pandas as pd

from corl.evaluation.episode_artifact import EpisodeArtifact


@dataclasses.dataclass
class EvaluationOutcome:
    """Dataclass to hold the results of an evaluation run
  """

    @dataclasses.dataclass
    class Dataframes:
        """Dataclass that holds dataframes generated from an EvaluationOutcome
    """
        step: pd.DataFrame
        episode: pd.DataFrame
        done: pd.DataFrame

    test_cases: typing.Union[pd.DataFrame, list, None]
    episode_artifacts: typing.Dict[int, EpisodeArtifact]
