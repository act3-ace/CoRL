"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Load an agent from a weight file
"""
import dataclasses
import logging
import pathlib
import time
import typing
import warnings

import h5py
from ray.rllib.algorithms import Algorithm

from corl.evaluation.loader.i_agent_loader import IAgentLoader
from corl.evaluation.loader.policy_checkpoint import PolicyCheckpoint


@dataclasses.dataclass
class WeightFile(IAgentLoader):
    """Class manages loading an agent from a weight file
    """

    h5_file_path: str

    def __post_init__(self):
        warnings.warn(f'{self.__class__.__name__} will be deprecated. Use {PolicyCheckpoint} instead', DeprecationWarning, stacklevel=2)

        self._log: logging.Logger = logging.getLogger(WeightFile.__name__)
        self._raw_weights: typing.OrderedDict[str, typing.Any] = {}

        weight_file = pathlib.Path(self.h5_file_path)
        if not weight_file.exists():
            raise FileNotFoundError(f"Policy weights file {self.h5_file_path} does not exist!")

        for attempt in range(3):
            try:
                # make our visit function
                def visit_items_func(name: str, obj: typing.Union[h5py.Group, h5py.Dataset]):
                    if isinstance(obj, h5py.Dataset):
                        self._raw_weights[name] = obj[()]  # pylint: disable=cell-var-from-loop

                # call our visit function on the h5 file
                with h5py.File(self.h5_file_path, "r") as f:
                    f.visititems(visit_items_func)
                break
            except OSError:  # pragma: no cover
                time.sleep(5)
                self._log.debug(f"Attempt number {attempt} failed to load policy from path {self.h5_file_path}")
                continue
        else:  # pragma: no cover
            raise OSError(f"Exceeded max attempts to read file: {self.h5_file_path}")

    def apply_to_algorithm(self, algorithm: Algorithm, policy_to_apply: str) -> None:
        """Applies to loaded weight file to an RLLIB algorithm

        Arguments:
            algorithm {Algorithm} -- Algorithm to apply weight file to
            policy_to_apply {str} -- Policy in the algorithm to apply weights to
        """

        IAgentLoader._apply_policy_to_weights(
            algorithm,
            self._raw_weights,
            policy_to_apply,
        )

        self._log.debug(f"Loaded weights into algorithm policy {policy_to_apply}")
