"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
policy mapping functions
"""
from abc import abstractmethod

from pydantic import BaseModel


class PolicyMappingValidatorBase(BaseModel):
    ...


class PolicyMappingBase:
    """Base class for policy mapping implementations"""

    def __init__(self, **kwargs) -> None:
        self.config = self.get_validator()(**kwargs)

    @staticmethod
    def get_validator() -> type[PolicyMappingValidatorBase]:
        """
        get validator for this Done Functor
        """
        return PolicyMappingValidatorBase

    @abstractmethod
    def __call__(self, agent_id, episode, worker):
        """
        policy mapping functions
        """
        raise NotImplementedError


class PolicyIsAgent(PolicyMappingBase):
    """Map all agents to same policy"""

    def __call__(self, agent_id, episode, worker):
        return agent_id


class SinglePolicyValidator(PolicyMappingValidatorBase):
    """SinglePolicyValidator"""

    policy_id: str = "blue0"


class SinglePolicy(PolicyMappingBase):
    """Map all agents to same policy"""

    def __init__(self, **kwargs) -> None:
        self.config: SinglePolicyValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[SinglePolicyValidator]:
        """
        get validator for this Done Functor
        """
        return SinglePolicyValidator

    def __call__(self, agentid, episode, worker):
        return self.config.policy_id
