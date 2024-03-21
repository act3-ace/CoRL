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
multiple actions like with platform and sensor control. These connections also
apply to observations for instance with multiple observations like platform
state and sensor observations. How the actions and observations are "glued"
together is specific to the task and thus require an implementation of the base
class.

The glue you use must match the actual agent in the environment or be robust
enough to handle that agent. For example if there are different controllers
than expected do you throw an error or just reduce the action space.
A good practice is to assert the control or sensor is a type or a
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
from functools import cached_property

import gymnasium
import tree
from pydantic import BaseModel, ConfigDict, ImportString

from corl.libraries.env_space_util import EnvSpaceUtil
from corl.libraries.normalization import LinearNormalizer, Normalizer
from corl.libraries.property import Prop
from corl.libraries.units import Quantity
from corl.rewards.base_measurement_operation import ExtractorSet, ObservationExtractorValidator
from corl.simulators.base_platform import BasePlatform


class TrainingExportBehavior(str, enum.Enum):
    """Enumeration of training behaviors"""

    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"


class BaseAgentGlueNormalizationValidator(BaseModel):
    """
    enabled: if normalization is enabled for this glue
    minimum: the minimum value this glue's output will be normalized to
    maximum: the maximum value this glue's output will be normalized to
    """

    enabled: bool = True
    normalizer: ImportString = LinearNormalizer
    config: dict = {}


class BaseAgentGlueValidator(BaseModel):
    """
    name: the custom name of this glue
    agent_name: the name of the agent who owns this glue
    seed: temporary - assume this will be removed
    normalization: how to handle obs normalization, see BaseAgentGlueNormalizationValidator
    training_export_behavior: if this glues output should be included in the obs provided to policy
    clip_to_space: if this glues output should be clipped to match the obs space
    extractors: A dict of extractors to initialize for use with the topologically sorted
                    obs dictionary
    """

    name: str | None = None
    agent_name: str
    seed: int | None = None
    normalization: BaseAgentGlueNormalizationValidator = BaseAgentGlueNormalizationValidator()
    training_export_behavior: TrainingExportBehavior = TrainingExportBehavior.INCLUDE
    clip_to_space: bool = False
    extractors: dict[str, ObservationExtractorValidator] = {}


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
        self.config: BaseAgentGlueValidator = self.get_validator()(**kwargs)
        self._normalizer: Normalizer = self.config.normalization.normalizer(**self.config.normalization.config)
        self._agent_removed = False
        # construct extractors
        self.extractors: dict[str, ExtractorSet] = {}
        for key, extractor in self.config.extractors.items():
            self.extractors[key] = extractor.construct_extractors()

        self._normalized_action_space: gymnasium.Space[typing.Any] | None = None
        self._normalized_observation_space: gymnasium.Space[typing.Any] | None = None

    @staticmethod
    def get_validator() -> type[BaseAgentGlueValidator]:
        """returns the validator for this class

        Returns:
            BaseAgentGlueValidator -- A pydantic validator to be used to validate kwargs
        """
        return BaseAgentGlueValidator

    @abc.abstractmethod
    def get_unique_name(self) -> str:
        """Provides a unique name of the glue to differentiate it from other glues."""

    @cached_property
    def action_prop(self) -> Prop | None:
        """
        Build the action property for the controllers that defines the action this glue produces

        Returns
        -------
        typing.Optional[Prop]
            The Property that defines what this glue requires for an action
        """

    @cached_property
    def action_space(self) -> gymnasium.Space | None:
        """
        Build the action space for the controllers that defines the action given to apply_action
        This property should almost never need to be overridden

        Returns
        -------
        gymnasium.spaces.Space
            The gymnasium Space that defines the action given to the apply_action function
        """
        return self.action_prop.create_space() if self.action_prop else None

    @cached_property
    def normalized_action_space(self) -> gymnasium.Space | None:
        """
        Normalizes an action space from this glue to some normalized bounds.
        There is not rules on what "normal" is.
        The only idea is that the "normal" range is what the Policy will output

        The default implementation scales all Box spaces to a low=-1. and high =1.

        This function should be a dual function to the unnormalize_action

        Returns
        -------
        gymnasium.spaces.Space:
            The scaled action space
        """
        if not self._normalized_action_space and (action_space := self.action_space):
            self._normalized_action_space = (
                self._normalizer.normalize_space(space=action_space) if self.config.normalization.enabled else action_space
            )
        return self._normalized_action_space

    def apply_action(
        self,
        action: EnvSpaceUtil.sample_type,
        observation: EnvSpaceUtil.sample_type,
        action_space: gymnasium.Space,
        obs_space: gymnasium.Space,
        obs_units: OrderedDict,
    ) -> None:
        """
        Apply the action for the controllers

        Parameters
        ----------
        action
            The action for the class to apply to the platform
        observation
            The current observations before applying the action
        action_space: OrderedDict
            The action space for this agent
        obs_space: OrderedDict
            The observation space for this agent
        obs_units:
            The observation units for this agent
        """

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
        ret: EnvSpaceUtil.sample_type = action
        if self.config.normalization.enabled:
            ret = self._normalizer.unnormalize_sample(space=self.action_space, sample=action)  # type: ignore
        return ret

    @cached_property
    @abc.abstractmethod
    def observation_prop(self) -> Prop:
        """
        Build the property descriving the observation space for
        the platform using the state of the platform, controller, sensors, etc.

        Returns
        -------
        Prop
            The corl Prop that defines the returned space from the get_observation function
        """

    @cached_property
    def observation_space(self) -> gymnasium.Space | None:
        """
        Build the observation space for the platform using the state of the platform, controller, sensors, etc.
        This property should almost never need to be overridden

        Returns
        -------
        gymnasium.spaces.Space
            The gymnasium space that defines the returned space from the get_observation function
        """
        return self.observation_prop.create_space() if self.observation_prop else None

    @cached_property
    def normalized_observation_space(self) -> gymnasium.Space | None:
        """
        Normalizes an observation space from this glue to some normalized bounds.
        There is not rules on what "normal" is.
        The only idea is that the "normal" range is what the Policy will take as input for observations

        The default implementation scales all Box spaces to a low=-1. and high =1.

        This function should be the same as the normalize_observation function but return a Space and not the sample

        Returns
        -------
        gymnasium.spaces.Space:
            The scaled observation space
        """
        if not self._normalized_observation_space and (observation_space := self.observation_space):
            if self.config.normalization.enabled:
                tmp = self._normalizer.normalize_space(space=observation_space)
                # if not isinstance(tmp, gymnasium.spaces.Dict):
                #     raise RuntimeError(
                #         f"{self.get_unique_name()} tried to return something other than a dict observation_space, "
                #         "this is currently not supported."
                #     )
                self._normalized_observation_space = tmp
            else:
                self._normalized_observation_space = observation_space
        return self._normalized_observation_space

    @abc.abstractmethod
    def get_observation(self, other_obs: OrderedDict, obs_space: gymnasium.Space, obs_units: OrderedDict) -> EnvSpaceUtil.sample_type:
        """
        Get the actual observation for the platform using the state of the platform, controller, sensors, etc.

        Parameters
        ----------
        other_obs: OrderedDict
            The observation dict containing any observations this glue may depend on
        obs_space: OrderedDict
            The observation space for this agent
        obs_units:
            The obserbation units for this agent

        Returns
        -------
        EnvSpaceUtil.sample_type
            The actual observation for this platform from this glue class
        """

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
        ret = tree.map_structure(lambda v: v.m if isinstance(v, Quantity) else v, observation)
        if self.config.normalization.enabled:
            obs_space = self.observation_space
            # if isinstance(obs_space, gymnasium.spaces.Dict) and any(isinstance(obs_space[key], Repeated) for key in obs_space.keys()):
            #     ret = self._normalizer.normalize_sample(space=obs_space, sample=copy.deepcopy(ret))
            # elif obs_space:
            ret = self._normalizer.normalize_sample(space=obs_space, sample=ret)  # type: ignore
        return ret

    def get_info_dict(self) -> EnvSpaceUtil.sample_type:  # noqa: PLR6301
        """
        Get the user specified metadata/metrics/etc.

        Returns
        -------
        EnvSpaceUtil.sample_type
            The actual info dict object for this platform from this glue class
        """
        return {}

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
    model_config = ConfigDict(arbitrary_types_allowed=True)


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

    @staticmethod
    def get_validator() -> type[BaseAgentPlatformGlueValidator]:
        return BaseAgentPlatformGlueValidator


class BaseAgentControllerGlue(BaseAgentPlatformGlue, abc.ABC):
    """
    BaseAgentControllerGlue assumes that this glue is for a controller and that an associated 'get_applied_control' method is available
    """

    @abc.abstractmethod
    def get_applied_control(self) -> Quantity:
        """
        Get the currently applied controls

        Returns
        -------
        Quantity
            The currently applied control
        """
