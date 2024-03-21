"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import dataclasses

import jsonargparse

from corl.evaluation.recording.i_recorder import IRecord


@jsonargparse.typing.final
@dataclasses.dataclass
class EvaluationArtifact_EvaluationOutcome:
    """
    Contains information needed to retrieve the evaluation outcome

    Args:
        location: Location of Evaluation outcome
    """

    location: IRecord


@jsonargparse.typing.final
@dataclasses.dataclass
class EvaluationArtifact_Metrics:
    """
    Contains information to describe artifacts from generating metrics
    Artifacts of metrics is a file stored on some location of the filesystem
    Later this may be extended to be connection to a database

    Args:
        location: Location of metrics artifact, currently only file system is supported
        metrics_file: Name of file to save metrics to. defaults to metrics.pkl
    """

    location: str  # Eventually this may be a reference to a database instead of just a path
    file: str = "metrics.pkl"


@jsonargparse.typing.final
@dataclasses.dataclass
class EvaluationArtifact_Visualization:
    """
    Contains information to describe visualization artifacts
    Artifacts of visualization are various files (depends on specific technique) stored on the filesystem
    Later this may be extended to be connection to a database

    Args:
        location: Location of plot artifacts
    """

    location: str  # Eventually this may be a reference to a database instead of just a path
