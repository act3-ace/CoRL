"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
This base class (BasePlatformGlue) is the "glue" that sticks the base
integration object(s) to the environment.This class and its derived classes
are expected to be agnostic to the specific integration. The idea is that
you can then use the same "glue" for training and inference on any
integration.

The main purpose of the code in this class is to expose the specific actions
and observations for your given task. For example you may not want the full
controls from a BaseController for a given problem. This class is where you
decide to modify the controls exposed to the policy to only include what you
want actually controlled.

Deciding on how to combine actions is much more important when there are
multiple actions like with movement. These connections also apply
to observations for instance with multiple observations like platform state
and sensor observations. How the actions and observations are "glued" together
is specific to the task and thus require an implementation of the base class.

The glue you use must match the actual agent in the environment or be robust
enough to handle that agent. For example if there are different controllers
than expected do you throw an error or just reduce the action space.
A good practice is to assert the control, or sensor is a type or a
subclass of a type to guarantee it has the functionality you expect.

The glue classes are modular in that you can use multiple glue classes per
agent. This allows stacking a glue to a stick controller glue to get
a and stick controlled agent. However, some control schemes may not be
compatible with each other. For example if you try to add a pitch control with
the stick and a pitch rate controller, the resulting behavior is not
straightforward and may cause your backend to throw an error.
"""
import abc
import enum
import logging
import typing
from collections import OrderedDict
from functools import lru_cache

import gym
from pydantic import BaseModel

from corl.libraries.env_space_util import EnvSpaceUtil
from corl.simulators.base_platform import BasePlatform


class TrainingExportBehavior(str, enum.Enum):
    """Enumeration of training behaviors
    """
    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"


class BaseAgentGlueNormalizationValidator(BaseModel):
    """
    enabled: if normalization is enabled for this glue
    minimum: the minimum value this glue's output will be normalized to
    maximum: the maximum value this glue's output will be normalized to
    """
    enabled: bool = True
    minimum: float = -1.0
    maximum: float = 1.0


class BaseAgentGlueValidator(BaseModel):
    """
    name: the custom name of this glue
    agent_name: the name of the agent who owns this glue
    seed: temporary - assume this will be removed
    normalization: how to handle obs normalization, see BaseAgentGlueNormalizationValidator
    """
    name: typing.Optional[str]
    agent_name: str
    seed: typing.Optional[int]
    normalization: BaseAgentGlueNormalizationValidator = BaseAgentGlueNormalizationValidator()
    training_export_behavior: TrainingExportBehavior = TrainingExportBehavior.INCLUDE


class BaseAgentGlue(abc.ABC):
    """
    BasePlatformGlue abstract class that provides the action space, observation space and how to apply actions and get observations
    for a platform

    """

    def __init__(self, **kwargs) -> None:
        """
        The init function for an Agent Glue class

        Parameters
        ----------
        state: typing.Tuple[BasePlatform]
            The initial state of the environment so the glue can stick together platform(s) or platform parts
        agent_id: str
            The name of the agent this glue is stuck to
        config: dict
            The configuration parameters of this glue class
        """
        self.config: BaseAgentGlueValidator = self.get_validator(**kwargs)
        self._agent_removed = False

    @property
    def get_validator(self) -> typing.Type[BaseAgentGlueValidator]:
        """returns the validator for this class

        Returns:
            BaseAgentGlueValidator -- A pydantic validator to be used to validate kwargs
        """
        return BaseAgentGlueValidator

    @lru_cache(maxsize=1)
    @abc.abstractmethod
    def get_unique_name(self) -> str:
        """Provies a unique name of the glue to differiciate it from other glues.
        """
        ...

    @lru_cache(maxsize=1)
    def action_space(self) -> gym.spaces.Space:
        """
        Build the action space for the controller, etc. that defines the action given to apply_action

        Returns
        -------
        gym.spaces.Space
            The gym Space that defines the action given to the apply_action function
        """
        ...

    @lru_cache(maxsize=1)
    def normalized_action_space(self) -> typing.Optional[gym.spaces.Space]:
        """
        Normalizes an action space from this glue to some normalized bounds.
        There is not rules on what "normal" is.
        The only idea is that the "normal" range is what the Policy will output

        The default implementation scales all Box spaces to a low=-1. and high =1.

        This function should be a dual function to the unnormalize_action

        Returns
        -------
        gym.spaces.Space:
            The scaled action space
        """
        action_space = self.action_space()
        if action_space and self.config.normalization.enabled:
            return EnvSpaceUtil.normalize_space(
                space=action_space, out_min=self.config.normalization.minimum, out_max=self.config.normalization.maximum
            )
        if action_space:
            return action_space
        return None

    def apply_action(
        self,
        action: EnvSpaceUtil.sample_type,  # pylint: disable=unused-argument
        observation: EnvSpaceUtil.sample_type  # pylint: disable=unused-argument
    ) -> None:
        """
        Apply the action for the controller, etc.

        Parameters
        ----------
        action
            The action for the class to apply to the platform
        observation
            The current observations before appling the action
        """
        ...

    def unnormalize_action(self, action: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        Un-Normalizes an action space from this glue's normalized_action_space to some raw bounds

        The default implementation assumes the normalized scale was low=-1. and high =1.

        This function should be a dual function to the normalize_action_space

        Parameters
        ----------
        action: EnvSpaceUtil.sample_type
            The action to unnormalize
        Returns
        -------
        EnvSpaceUtil.sample_type:
            the unnormalized action
        """
        if self.config.normalization.enabled:
            ret = EnvSpaceUtil.unscale_sample_from_space(
                space=self.action_space(),
                space_sample=action,
                out_min=self.config.normalization.minimum,
                out_max=self.config.normalization.maximum
            )
        else:
            ret = action
        return ret

    @lru_cache(maxsize=1)
    def observation_space(self) -> gym.spaces.Space:
        """
        Build the observation space for the platform using the state of the platform, controller, sensors, etc.

        Returns
        -------
        gym.spaces.Space
            The gym space that defines the returned space from the get_observation function
        """
        ...

    @lru_cache(maxsize=1)
    def normalized_observation_space(self) -> typing.Optional[gym.spaces.Space]:
        """
        Normalizes an observation space from this glue to some normalized bounds.
        There is not rules on what "normal" is.
        The only idea is that the "normal" range is what the Policy will take as input for observations

        The default implementation scales all Box spaces to a low=-1. and high =1.

        This function should be the same as the normalize_observation function but return a Space and not the sample

        Returns
        -------
        gym.spaces.Space:
            The scaled observation space
        """
        observation_space = self.observation_space()
        if observation_space and self.config.normalization.enabled:
            return EnvSpaceUtil.normalize_space(
                space=observation_space, out_min=self.config.normalization.minimum, out_max=self.config.normalization.maximum
            )
        if observation_space:
            return observation_space
        return None

    def get_observation(self) -> EnvSpaceUtil.sample_type:
        """
        Get the actual observation for the platform using the state of the platform, controller, sensors, etc.

        Returns
        -------
        EnvSpaceUtil.sample_type
            The actual observation for this platform from this glue class
        """
        ...

    def normalize_observation(self, observation: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        Normalizes an observation from this glue to some normalized bounds.
        There is not rules on what "normal" is.
        The only idea is that the "normal" range is what the Policy will use as observations

        Parameters
        ----------
        observation: EnvSpaceUtil.sample_type
            The observation we want to scale
        Returns
        -------
        EnvSpaceUtil.sample_type:
            The scaled observation
        """
        if not self.config.normalization.enabled:
            ret = observation
        else:
            ret = EnvSpaceUtil.scale_sample_from_space(
                space=self.observation_space(),
                space_sample=observation,
                out_min=self.config.normalization.minimum,
                out_max=self.config.normalization.maximum,
            )
        return ret

    def get_info_dict(self) -> EnvSpaceUtil.sample_type:
        """
        Get the user specified metadata/metrics/etc.

        Returns
        -------
        EnvSpaceUtil.sample_type
            The actual info dict object for this platform from this glue class
        """
        ...

    def agent_removed(self) -> bool:
        """
        Returns true if the agent has been removed, false otherwise
        """
        return self._agent_removed

    def set_agent_removed(self, agent_removed: bool = True) -> None:
        """
        Notify the glue that the agent it is 'attached' to has been removed by the simulation
        """
        self._agent_removed = agent_removed


class BaseAgentPlatformGlueValidator(BaseAgentGlueValidator):
    """
    platform: The platform object that this glue will read from/apply actions to
    """
    platform: BasePlatform

    class Config:
        """
        This allows pydantic to validate that platform is a BasePlatform
        """
        arbitrary_types_allowed = True


class BaseAgentPlatformGlue(BaseAgentGlue, abc.ABC):
    """
    BaseAgentPlatformGlue assumes that this glue corresponds to exactly one platform with the same
    platform name as the agent_id this glue is attached to

    This will then set the self._platform to this platform
    """

    def __init__(self, **kwargs) -> None:
        self.config: BaseAgentPlatformGlueValidator
        super().__init__(**kwargs)
        self._platform: BasePlatform = self.config.platform
        self._logger = logging.getLogger(BaseAgentPlatformGlue.__name__)

    @property
    def get_validator(self) -> typing.Type[BaseAgentPlatformGlueValidator]:
        return BaseAgentPlatformGlueValidator


class BaseAgentControllerGlue(BaseAgentPlatformGlue, abc.ABC):
    """
    BaseAgentControllerGlue assumes that this glue is for a controller and that an associated 'get_applied_control' method is available
    """

    @abc.abstractmethod
    def get_applied_control(self) -> OrderedDict:
        """
        Get the currently applied controls

        Returns
        -------
        OrderedDict
            The currently applied controls
        """
        ...
