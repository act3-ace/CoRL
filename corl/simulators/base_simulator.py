"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import typing
from abc import ABC, abstractmethod

from pydantic import BaseModel, ImportString, TypeAdapter, ValidationError, validate_call
from ray.rllib import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

from corl.libraries.factory import Factory
from corl.simulators.base_platform import BasePlatform
from corl.simulators.base_simulator_state import BaseSimulatorState


class AgentConfig(BaseModel):
    """
    platform_config: any configuration needed for the simulator to
                initialize this platform and configure it in the sim class
    parts_list: a list of tuples where the first element is come python class path
                    of a BasePart, and then the second element is a configuration dictionary for that part

    Arguments:
        BaseModel: [description]
    """

    platform_config: dict[str, typing.Any] | BaseModel
    parts_list: list[tuple[ImportString, dict[str, typing.Any]]]


class BaseSimulatorValidator(BaseModel):
    """
    worker_index: what worker this simulator class is running on < used for render
    vector_index: what vector index this simulator class is running on < used for render
    agent_configs: the mapping of agent names to a dict describing the platform
    frame_rate: the rate the simulator should run at (in Hz)
    """

    worker_index: int = 0
    vector_index: int = 0
    agent_configs: typing.Mapping[str, AgentConfig]
    frame_rate: float = 1.0


class BaseSimulatorResetValidator(BaseModel):
    """
    Validator to use to validate the reset input to a simulator class
    allows the simulator class to take EPP params and structure/validate them

    Note that all attributes in this validator need to survive being parsed by validation_helper_units_and_parameters.

    Subclasses can redefine `platforms` to make the `typing.Any` more restrictive.  It must remain a dictionary with keys named for
    the platforms in the simulation.

    agent_configs_reset: the mapping of agent names to a dict describing the platform to be used during sim reset
    """

    platforms: dict[str, typing.Any] = {}
    agent_configs_reset: typing.Mapping[str, AgentConfig] = {}


@validate_call
def validation_helper_units_and_parameters(value: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Recursively inspect a dictionary, converting Factory"""

    output: dict[str, typing.Any] = {}
    for k, v in value.items():
        try:
            factory = TypeAdapter(Factory).validate_python(v)
        except ValidationError:
            pass
        else:
            output[k] = factory.build()
            continue

        try:
            output[k] = validation_helper_units_and_parameters(v)
        except ValidationError:
            pass
        else:
            continue

        output[k] = v

    return output


class SimulatorDefaultCallbacks:
    """This class allows the simulator to add simulator specific operation into the environment callbacks.

    These methods are called at the end of corl.environment.default_env_rllib_callbacks.EnvironmentDefaultCallbacks methods of the same
    name.
    """

    def on_episode_start(
        self,
        *,
        worker,
        base_env: BaseEnv,
        policies: dict[PolicyID, Policy],
        episode: EpisodeV2,
        **kwargs,
    ) -> None:
        """Callback run on the rollout worker before each episode starts.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode: EpisodeV2 object which contains the episode's
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """

    def on_episode_step(
        self,
        *,
        worker,
        base_env: BaseEnv,
        policies: dict[PolicyID, Policy] | None = None,
        episode: EpisodeV2,
        **kwargs,
    ) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: EpisodeV2 object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """

    def on_episode_end(self, *, worker, base_env: BaseEnv, policies: dict[PolicyID, Policy], episode, **kwargs) -> None:
        """
        Runs at the end of the episode.

        Parameters
        ----------
        worker: RolloutWorker
            Reference to the current rollout worker.
        base_env: BaseEnv
            BaseEnv running the episode. The underlying
            env object can be gotten by calling base_env.get_sub_environments().
        policies: dict
            Mapping of policy id to policy objects. In single
            agent mode there will only be a single "default" policy.
        episode: MultiAgentEpisode
            EpisodeV2 object which contains episode
            state. You can use the `episode.user_data` dict to store
            temporary data, and `episode.custom_metrics` to store custom
            metrics for the episode.
        """


class BaseSimulator(ABC):
    """
    BaseSimulator is responsible for initializing the platform objects for a simulation
    and knowing how to setup episodes based on input parameters from a parameter provider
    it is also responsible for reporting the simulation state at each timestep
    """

    def __init__(self, **kwargs) -> None:
        self.config = self.get_simulator_validator(**kwargs)
        self.callbacks = self.get_callbacks()

    @property
    def get_simulator_validator(self) -> type[BaseSimulatorValidator]:
        """
        returns the validator for the configuration options to the simulator
        the kwargs to this class are validated and put into a defined struct
        potentially raising based on invalid configurations

        Returns:
            BaseSimulatorValidator -- The validator to use for this simulation class
        """
        return BaseSimulatorValidator

    @property
    def get_reset_validator(self) -> type[BaseSimulatorResetValidator]:
        """
        returns the validator that can be used to validate episode parameters
        coming into the reset function from the environment class

        Returns:
            BaseSimulatorResetValidator -- The validator to use during resets
        """
        return BaseSimulatorResetValidator

    @property
    def frame_rate(self) -> float:
        """Return the frame rate (in Hz) this simulator will run at"""
        return self.config.frame_rate

    def shutdown(self):
        """Shutdown the simulator. This will cause all resources to be closed. After this is called,
        reset/step may no longer be called and the simulation is ready for destruction. This should be
        safe to be called more than once, though subsequent calls should have no effect.
        """

    @abstractmethod
    def reset(self, config: dict[str, typing.Any]) -> BaseSimulatorState:
        """
        reset resets the simulation and sets up a new episode

        Arguments:
            config {typing.Dict[str, typing.Any]} -- The parameters to
                    validate and use to setup this episode

        Returns:
            BaseSimulatorState -- The simulation state
        """

    @abstractmethod
    def step(self, platforms_to_action: set[str]) -> BaseSimulatorState:
        """
        advances the simulation platforms and returns the state

        Arguments:
            platforms_to_action {typing.Dict[str, typing.Any]} -- the List of platform names who the
                environment believes should perform a new action on this call to step

        Returns:
            BaseSimulatorState -- The state after the simulation updates
        """

    @property
    @abstractmethod
    def sim_time(self) -> float:
        """
        returns the time

        Returns:
            float - time
        """

    @property
    @abstractmethod
    def platforms(self) -> dict[str, BasePlatform]:
        """
        returns a dict of platforms in the simulation

        Returns:
            dict of platforms mapped platform_name -> platform
        """

    @abstractmethod
    def mark_episode_done(self, done_info: dict, episode_state: dict, metadata: dict | None = None):
        """
        Takes in the done_info specifying how the episode completed
        and does any book keeping around ending an episode

        Arguments:
            done_info {OrderedDict} -- The Dict describing which Done conditions ended an episode
            episode_state {OrderedDict} -- The episode state at the end of the simulation
            metadata {dict | None}  --  Additional metadata to be stored along with the episode
        """

    @abstractmethod
    def save_episode_information(self, dones, rewards, observations, observation_units):
        """
        provides a way to save information about the current episode
        based on the environment

        Arguments:
            dones: the current done info of the step
            rewards: the reward info for this step
            observations: the observations for this step
            observation_units: the measurement units of the observations for this step
        """

    @property
    def get_callbacks(self) -> type[SimulatorDefaultCallbacks]:
        """Return the simulator callback class."""
        return SimulatorDefaultCallbacks

    def render(self, state, mode="human"):
        """
        allows you to do something to render your simulation
        you are responsible for checking which worker/vector index you are on
        """

    def delete_platform(self, name):
        """
        provides a way to delete a platform from the simulation
        """
