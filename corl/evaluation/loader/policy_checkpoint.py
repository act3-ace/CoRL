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
import os
import typing
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.policy import Policy

from corl.evaluation.loader.i_agent_loader import IAgentLoader


class PolicyCheckpoint(BaseModel, IAgentLoader):
    """Class manages loading an agent from a checkpoint file
    Because this is used/instantiated via jsonargparse, we need to use pydantic.dataclasses.dataclass here (instead of BaseModel);
    in conjunction with jsonargparse, this results in some (unfortunate) side effects:
      1. Can't directly use Path
      2. Can't call validators before some validation is performed (i.e. @validator `pre` parameter doesn't behave as expected)
        a. The primary side effect of this is that derived values must be defined as Optional, even though they will exist after validation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trained_agent_id: str | None = None
    checkpoint_filename: Path

    policy_: Policy | None = None

    @field_validator("checkpoint_filename", mode="before")
    def validate_checkpoint_filename(cls, v, values):
        """Validates the checkpoint directory"""
        assert v is not None, "checkpoint_filename must be set"
        p = Path(os.path.expandvars(str(v)))
        assert p.exists(), f"{p} does not exist"
        assert p.is_dir(), f"{p} must be a directory"

        trained_agent_id = values.data["trained_agent_id"] if values.data["trained_agent_id"] is not None else ""

        algorithm_policy_state_file = p / "policies" / trained_agent_id / "policy_state.pkl"
        policy_policy_state_file = p / "policy_state.pkl"

        if not (algorithm_policy_state_file.exists() or policy_policy_state_file.exists()):
            raise ValueError(f"No policy state found for {p}:\n\t{algorithm_policy_state_file}\n\t{policy_policy_state_file}")

        return p

    @property
    def policy(self) -> Policy:
        """
        returns the policy object given the class config

        Raises:
            RuntimeError: if the specified agent is not the checkpoint

        Returns:
            Policy: The policy from the checkpoint
        """
        if not self.policy_:
            policy = Policy.from_checkpoint(str(self.checkpoint_filename))

            if isinstance(policy, dict):
                if self.agent_id not in policy:
                    raise RuntimeError(f"agent_id '{self.agent_id}' not in '{policy}'")
                self.policy_ = policy[self.agent_id]
            elif isinstance(policy, Policy):
                self.policy_ = policy
            else:
                raise TypeError(f"Unknown type {type(policy)} for {Policy}")

        return typing.cast(Policy, self.policy_)

    @property
    def was_trained(self) -> bool:
        """
        specifies if a policy to be extract was trained in the checkpoint
        """
        return self.agent_id in self.policy.config["multiagent"]["policies_to_train"]

    @property
    def agent_id(self) -> str | None:
        if self.trained_agent_id is not None:
            return self.trained_agent_id

        if "agent_id" not in self.policy.config:
            return Path(self.checkpoint_filename).name

        return self.policy.config["agent_id"]

    @property
    def env_config(self) -> dict | None:
        return self.policy.config["env_config"]

    def apply_to_algorithm(self, algorithm: Algorithm, policy_to_apply: str) -> None:
        """Applies the weights from checkpoint file to an RLLIB algorithm

        Arguments:
            algorithm {Algorithm} -- Algorithm to apply weight file to
            policy_to_apply {str} -- Policy in the algorithm to apply weights to
        """
        if weights := self.policy.get_weights():
            self._apply_policy_to_weights(
                algorithm=algorithm,
                weights=weights,
                policy_to_apply=policy_to_apply,
            )
        else:
            raise RuntimeError(f"Policy {self.checkpoint_filename} was not trained - No weights available.")
