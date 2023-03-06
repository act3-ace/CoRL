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
import dataclasses
import os
import shutil
import typing
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import ray.cloudpickle as pickle

from corl.evaluation.evaluation_outcome import EvaluationOutcome
from corl.evaluation.recording.i_recorder import IRecord, IRecorder


class FolderRecord(IRecord):
    """A folder record which can save and load EvaluationOutcomes to folders
    """
    _absolute_path: Path

    def __init__(self, absolute_path: typing.Union[Path, str]):
        self._absolute_path = Path(absolute_path)

    @property
    def absolute_path(self) -> Path:
        """Get the absolute path
        """
        return self._absolute_path

    def save(self, outcome: EvaluationOutcome) -> None:
        """Save an evaluation outcome to a folder

        Arguments:
            outcome {EvaluationOutcome} -- Evaluation outcome to save
        """
        output_folder: Path = self._absolute_path

        for test_case_idx in outcome.episode_artifacts.keys():

            # retrieve current episode
            episode_artifact = outcome.episode_artifacts[test_case_idx]

            # create test case folder
            test_case_folder = output_folder / f"test_case_{str(test_case_idx)}"
            os.mkdir(test_case_folder)

            # essentially loop through the self.artifacts_filenames

            for artifact_name in episode_artifact.artifacts_filenames:
                # do relinking here , s.t. if an aer file or info file is present

                file_name = episode_artifact.artifacts_filenames[artifact_name].split('/')[-1]
                new_path = test_case_folder / file_name

                try:
                    copyfile(episode_artifact.artifacts_filenames[artifact_name], new_path)
                except FileNotFoundError as e:
                    print(f"{FolderRecord.__name__}:{e}")

            with open(test_case_folder / "batch.pkl", "wb") as f:
                pickle.dump(episode_artifact, f)

        with open(output_folder / "test_cases.pkl", "wb") as f:
            pickle.dump(outcome.test_cases, f)

    def load(self) -> EvaluationOutcome:
        """Load EvaluationOutcome from folder

        Returns:
            EvaluationOutcome -- Evaluation Outcome loaded from folder
        """

        with open(self._absolute_path / "test_cases.pkl", "rb") as f:
            test_cases = pickle.load(f)

        paths = self._absolute_path.glob('test_case_*/batch.pkl')
        artifacts = {}
        for path in paths:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
                if artifact.test_case in artifacts:
                    raise RuntimeError(
                        f"Test case of {artifact.test_case} already exists. This indicates an issue with evaluation generation"
                    )
                artifacts[artifact.test_case] = artifact

        return EvaluationOutcome(test_cases, artifacts)


@dataclasses.dataclass
class Folder(IRecorder[FolderRecord]):
    """Generates a record to interface with a folder structure
    """

    dir: str
    append_timestamp: bool = dataclasses.field(default=True)

    def resolve(self) -> FolderRecord:
        """Resolve and generate the record

        Returns:
            FolderRecord -- Record
        """

        #########################################
        ## Resolve and create directory

        output_folder = Path(self.dir)

        if self.append_timestamp:
            timestamp_str = '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_folder = output_folder.with_name(output_folder.stem + timestamp_str)

        # MTB Removed - How do you use the checkpoint when there are multiple checkpoints?!
        # # If output_folder is not provided, build it from other information
        # if output_folder is None:
        #     output_folder_is_checkpoint_dir = False

        #     # Determine the root of the output folder
        #     # First choice - output_root parameter
        #     output_folder = config['output_root']
        #     # Second choice - checkpoint directory
        #     if output_folder is None and config['checkpoint'] is not None:
        #         output_folder = os.path.dirname(config['checkpoint'])
        #         output_folder_is_checkpoint_dir = True
        #     # Third choice - parameters file directory
        #     if output_folder is None and config['parameters'] is not None:
        #         output_folder = os.path.dirname(config['parameters'])
        #         output_folder_is_checkpoint_dir = True
        #     # Fourth choice - current working directory
        #     if output_folder is None:
        #         output_folder = ''

        #     # Determine the name
        #     # First choice - name parameter
        #     name = self.name

        #     # Second choice - read environment and controller from parameters
        #     # Use checkpoint as final portion if it exists
        #     if name is None and
        #           not output_folder_is_checkpoint_dir and
        #           (config['checkpoint'] is not None or config['parameters'] is not None):
        #         if config['checkpoint'] is not None:
        #             checkpoint_folder = os.path.basename(os.path.dirname(os.path.dirname(config['checkpoint'])))
        #             checkpoint_name = os.path.basename(config['checkpoint'])
        #         else:
        #             checkpoint_folder = ''
        #             checkpoint_name = 'evaluation'

        #         parameters = config['parameters']
        #         if parameters is None:
        #             parameters = os.path.join(os.path.dirname(os.path.dirname(config['checkpoint'])), 'params.pkl')

        #         with open(parameters, 'rb') as f:
        #             env_config = pickle.load(f)

        #         aaco_environment = env_config['env_config']['environment']['aaco_environment']
        #         controller = env_config['env_config']['environment']['TrialName']

        #         # Modify leaderboard.py to accept
        #         name = os.path.join(aaco_environment, controller, checkpoint_folder, checkpoint_name)

        #     # Third choice - evaluation
        #     if name is None:
        #         name = 'evaluation'

        #     output_folder = os.path.join(output_folder, name + timestamp_str)

        if output_folder.exists():
            shutil.rmtree(output_folder)

        # Make absolute
        output_folder_abs = output_folder.resolve()

        output_folder_abs.mkdir(parents=True, exist_ok=False)

        return FolderRecord(output_folder_abs)
