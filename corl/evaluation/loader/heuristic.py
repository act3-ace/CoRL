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

from ray.rllib.algorithms import Algorithm

from corl.evaluation.loader.i_agent_loader import IAgentLoader


@dataclasses.dataclass
class Heuristic(IAgentLoader):
    """An agent loader that makes no manipulation to a given algorithm"""

    def __post_init__(self) -> None:
        self._log: logging.Logger = logging.getLogger(Heuristic.__name__)

    def apply_to_algorithm(self, algorithm: Algorithm, policy_to_apply: str) -> None:
        """Do nothing with the algorithm given"""
