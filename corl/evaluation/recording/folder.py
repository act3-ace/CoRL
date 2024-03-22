"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Records an evaluation outcome to a folder
"""
import contextlib
import logging
import shutil
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import pandas as pd
import ray.cloudpickle as pickle
from pydantic import BaseModel, field_validator

from corl.evaluation.evaluation_outcome import EvaluationOutcome
from corl.evaluation.recording.i_recorder import IRecord, IRecorder


class FolderRecord(IRecord):
    """A folder record which can save and load EvaluationOutcomes to folders"""

    _absolute_path: Path

    def __init__(self, absolute_path: Path | str):
        self._absolute_path = Path(absolute_path)
        self._logger = logging.getLogger(type(self).__name__)

    @property
    def absolute_path(self) -> Path:
        """Get the absolute path"""
        return self._absolute_path

    def save(self, outcome: EvaluationOutcome) -> None:
        """Save an evaluation outcome to a folder

        Arguments:
            outcome {EvaluationOutcome} -- Evaluation outcome to save
        """
        output_folder: Path = self._absolute_path

        for test_case_idx, episode_artifacts in outcome.episode_artifacts.items():
            # create test case folder
            test_case_folder = output_folder / f"test_case_{test_case_idx!s}"
            test_case_folder.mkdir(exist_ok=True)

            # Copy episode artifact files to test_case_folder
            for episode_artifact in episode_artifacts:
                start_time: str = episode_artifact.start_time.strftime("%Y-%m-%d_%H-%M-%S")

                for artifact_filename in episode_artifact.artifacts_filenames.values():
                    artifact_path = Path(artifact_filename)
                    new_path = test_case_folder / f"{start_time}_{artifact_path.name}"

                    try:
                        copyfile(artifact_path, new_path)
                    except FileNotFoundError as e:
                        self._logger.error(f"{e}")

                with open(test_case_folder / f"{start_time}_episode_artifact.pkl", "wb") as f:
                    # HACK todo fix
                    tmp_done_config = None
                    with contextlib.suppress(AttributeError):
                        tmp_done_config = episode_artifact.done_config
                        del episode_artifact.done_config

                    with contextlib.suppress(KeyError):
                        del episode_artifact.env_config["epp_registry"]

                    pickle.dump(episode_artifact, f)
                    if tmp_done_config is not None:
                        episode_artifact.done_config = tmp_done_config

            with open(test_case_folder / f"{start_time}_test_case.csv", "w") as f:
                outcome.get_test_cases().iloc[test_case_idx].to_csv(f)

        with open(output_folder / "test_cases.pkl", "wb") as f:
            pickle.dump(outcome.test_cases, f)

        with open(output_folder / "test_cases.csv", "w") as f:
            outcome.get_test_cases().to_csv(f)

    def load(self) -> EvaluationOutcome:
        """Load EvaluationOutcome from folder

        Returns:
            EvaluationOutcome -- Evaluation Outcome loaded from folder
        """

        with open(self._absolute_path / "test_cases.pkl", "rb") as f:
            test_cases = pickle.load(f)

        paths = self._absolute_path.glob("test_case_*/*_episode_artifact.pkl")
        artifacts = {}
        for path in paths:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
                if artifact.test_case in artifacts:
                    raise RuntimeError(
                        f"Test case of {artifact.test_case} already exists. This indicates an issue with evaluation generation"
                    )
                artifacts[artifact.test_case] = [artifact]

        return EvaluationOutcome(test_cases, artifacts)


class Folder(BaseModel, IRecorder[FolderRecord]):
    """Generates a record to interface with a folder structure"""

    append_timestamp: bool = True
    dir: Path  # noqa: A003

    @field_validator("dir")
    def generate_output_dir(cls, v, info):
        """Generate the output dir"""

        if info.data["append_timestamp"]:
            timestamp_str = "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_folder = v.with_name(v.stem + timestamp_str)
        else:
            output_folder = v

        if output_folder.exists():
            shutil.rmtree(str(output_folder))

        # Make absolute
        output_folder = output_folder.resolve()
        output_folder.mkdir(parents=True, exist_ok=False)
        return output_folder

    def resolve(self) -> FolderRecord:
        """Resolve and generate the record

        Returns:
            FolderRecord -- Record
        """
        return FolderRecord(self.dir)


class EpisodeArtifactLoggingCallbackLoader(IRecord):
    """Loads files saved from enabling the EpisodeArtifactLoggingCallback during
    training"""

    def __init__(self, absolute_path: Path | str):
        """_summary_

        Parameters
        ----------
        absolute_path : typing.Union[Path, str]
            This is the absolute path to the
        """

        self._absolute_path = Path(absolute_path)

    @property
    def absolute_path(self) -> Path:
        """Get the absolute path to the root directory where the trajectories
        are stored. (e.g. <experiment_name>/trajectories/epoch_000001)
        """

        return self._absolute_path

    def save(self, _):
        """Overwrite the abstract class"""

    def load(self) -> EvaluationOutcome:
        episode_artifact_files = list(self._absolute_path.glob("*.p*kl*"))
        episode_artifact_files.sort()
        artifacts_dict = {}
        # The test_cases will store initial conditions for each episode
        # artifact
        test_cases = []
        for idx, epi_artifact_file in enumerate(episode_artifact_files):
            with open(epi_artifact_file, "rb") as file_obj:
                episode_artifact = pickle.load(file_obj)
                # The episode artifact test case number doesn't matter
                # for artifacts saved from training
            episode_artifact.test_case = idx
            artifacts_dict[idx] = episode_artifact

            # Build the test case dataframe: the test cases dataframe stores the reset parameters
            # we grab the information from the episode artifact
            sim_reset_dict = episode_artifact.params
            # Add prefix so the columns match with what gets generated during IterateTestCases
            sim_reset_dict = {f"environment.{k}": v for k, v in sim_reset_dict.items()}
            sim_reset_dict["test_case"] = idx
            test_cases.append(sim_reset_dict)
        if test_cases:
            test_cases_df = pd.DataFrame(test_cases)
            test_cases_df = test_cases_df.set_index("test_case")

            return EvaluationOutcome(test_cases=test_cases_df, episode_artifacts=artifacts_dict)
        return EvaluationOutcome(test_cases=test_cases, episode_artifacts=artifacts_dict)
