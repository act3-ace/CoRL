"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import pathlib
import pickle
from abc import abstractmethod
from pathlib import Path

from corl.evaluation.evaluation_artifacts import EvaluationArtifact_Metrics, EvaluationArtifact_Visualization
from corl.evaluation.scene_processors import SceneProcessors


class Visualization:
    """Base class for a visualization"""

    def __init__(self):
        pass

    def load(self, metrics_location: EvaluationArtifact_Metrics, visualization_location: EvaluationArtifact_Visualization):
        """
        Load a config detailing the location of the post_processed_data location and the save location for visualizations

        Parameters
        ----------
        config: dict
            dictionary detailing location of post_processed data and save location

        """
        if isinstance(metrics_location.location, str):
            post_processed_data_location = pathlib.Path(metrics_location.location).joinpath(metrics_location.file)
            with open(post_processed_data_location, "rb") as fp:
                self._post_processed_data: SceneProcessors = pickle.load(fp)  # noqa: S301
        else:
            raise RuntimeError("Pulling metrics from anything other than folder not supported")

        if isinstance(visualization_location.location, str):
            self.out_folder = Path(visualization_location.location)
        else:
            raise RuntimeError("Saving visualization to anything aside folder not supported")

    @abstractmethod
    def visualize(self):
        """Execute visualization code"""
