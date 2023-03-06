"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Load and agent from Checkpoint file
"""
import dataclasses
import pathlib
import pickle
import warnings

from ray.rllib.algorithms import Algorithm

from corl.evaluation.loader.i_agent_loader import IAgentLoader
from corl.evaluation.loader.policy_checkpoint import PolicyCheckpoint


@dataclasses.dataclass
class CheckpointFile(IAgentLoader):
    """Class manages loading an agent from a checkpoint file
    checkpoint_filename: str
    """
    checkpoint_filename: str

    def __post_init__(self):
        warnings.warn(f'{self.__class__.__name__} will be deprecated. Use {PolicyCheckpoint} instead', DeprecationWarning, stacklevel=2)

        self._checkpoint_data = None

        checkpoint_file = pathlib.Path(self.checkpoint_filename)
        if not checkpoint_file.exists():
            raise FileNotFoundError(checkpoint_file)

        with open(checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)

        if "weights" not in checkpoint_data:
            raise RuntimeError(
                "There is no weights key in the checkpoint you provided, this means either your "
                "checkpoint is invalid, or it is a different checkpoint version than this code was written for"
                "this code was written based on version V1.0 checkpoints, and your version is: "
                f"{checkpoint_data['checkpoint_version']}"
            )

        self._checkpoint_data = checkpoint_data

    def apply_to_algorithm(self, algorithm: Algorithm, policy_to_apply: str) -> None:
        """Applies the weights from checkpoint file to an RLLIB algorithm

        Arguments:
            algorithm {Algorithm} -- Algorithm to apply weight file to
            policy_to_apply {str} -- Policy in the algorithm to apply weights to
        """

        IAgentLoader._apply_policy_to_weights(
            algorithm=algorithm,
            weights=self._checkpoint_data['weights'],
            policy_to_apply=policy_to_apply,
        )
