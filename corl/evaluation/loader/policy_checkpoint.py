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
import pathlib
import typing

from pydantic import dataclasses, validator
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.policy import Policy

from corl.evaluation.loader.i_agent_loader import IAgentLoader


class Config:
    """Pydantic options"""
    arbitrary_types_allowed = True


@dataclasses.dataclass(config=Config)
class PolicyCheckpoint(IAgentLoader):
    """Class manages loading an agent from a checkpoint file
    Because this is used/instantiated via jsonargparse, we need to use pydantic.dataclasses.dataclass here (instead of BaseModel);
    in conjunction with jsonargparse, this results in some (unfortunate) side effects:
      1. Can't directly use pathlib.Path
      2. Can't call validators before some validation is performed (i.e. @validator `pre` parameter doesn't behave as expected)
        a. The primary side effect of this is that derived values must be defined as Optional, even though they will exist after validation
    """
    checkpoint_filename: typing.Union[str, pathlib.Path]
    trained_agent_id: typing.Optional[str] = None

    _policy: typing.Optional[Policy] = None

    @validator('checkpoint_filename')
    def validate_checkpoint_filename(cls, v):
        """Validates the checkpoint directory"""
        p = pathlib.Path(str(v))
        assert p.exists(), f'{p} does not exist'
        assert p.is_dir(), f'{p} must be a directory'
        assert (p / 'policy_state.pkl').exists(), f"No file named 'policy_state.pkl' in {p}"
        return p

    @validator('trained_agent_id')
    def validate_trained_agent_id(cls, v, values):
        """Gets the agent_id used during training from the name of the directory"""
        if v:
            return v
        return values['checkpoint_filename'].name

    @property
    def policy(self) -> Policy:
        if not self._policy:
            policy = Policy.from_checkpoint(str(self.checkpoint_filename))
            agent_to_policies = list(policy.config['multiagent']['policies'].keys())
            if self.trained_agent_id not in agent_to_policies:
                raise RuntimeError(f"agent_id '{self.trained_agent_id}' not in '{agent_to_policies}'")
            self._policy = policy
        return typing.cast(Policy, self._policy)

    @property
    def was_trained(self) -> bool:
        return self.trained_agent_id in self.policy.config['multiagent']['policies_to_train']  # type: ignore

    @property
    def agent_id(self) -> typing.Optional[str]:
        return self.trained_agent_id

    @property
    def env_config(self) -> typing.Optional[typing.Dict]:
        return self.policy.config['env_config']  # type: ignore

    def apply_to_algorithm(self, algorithm: Algorithm, policy_to_apply: str) -> None:
        """Applies the weights from checkpoint file to an RLLIB algorithm

        Arguments:
            algorithm {Algorithm} -- Algorithm to apply weight file to
            policy_to_apply {str} -- Policy in the algorithm to apply weights to
        """
        if not self.was_trained:
            raise RuntimeError(f'Policy {self.checkpoint_filename} was not trained - No weights available.')

        IAgentLoader._apply_policy_to_weights(
            algorithm=algorithm,
            weights=self.policy.get_weights(),  # type: ignore
            policy_to_apply=policy_to_apply,
        )
