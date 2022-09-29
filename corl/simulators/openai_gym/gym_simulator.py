"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Base Simulator and Platform for Toy Openai Environments
This mainly shows a "how to use example" and provide an setup to unit test with
"""
import typing

import gym
from pydantic import BaseModel, PyObject

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.state_dict import StateDict
from corl.simulators.base_parts import BaseController, BaseSensor, MutuallyExclusiveParts
from corl.simulators.base_platform import BasePlatform, BasePlatformValidator
from corl.simulators.base_simulator import AgentConfig, BaseSimulator, BaseSimulatorValidator


class GymPlatformValidator(BasePlatformValidator):
    """GymPlatformValidator

    Parameters
    ----------
        platform: gym.Env
            Gym env associated with GymPlatform
    """
    platform: gym.Env


class GymPlatformConfig(BaseModel):
    """
    GymPlatformSimConfig
    """
    platform_class: PyObject


class GymAgentConfig(AgentConfig):
    """
    platform_config: any configuration needed for the simulator to
                initialize this platform and configure it in the sim class
    parts_list: a list of tuples where the first element is come python class path
                    of a BasePart, and then the second element is a configuration dictionary for that part

    Arguments:
        BaseModel {[type]} -- [description]
    """
    platform_config: GymPlatformConfig


class OpenAiGymPlatform(BasePlatform):
    """
    The OpenAiGymPlatform wraps some gym environment as it's platform and
    allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function
    """

    def __init__(self, **kwargs):
        kwargs["exclusive_part_dict"] = {
            BaseController: MutuallyExclusiveParts({"main_controller"}), BaseSensor: MutuallyExclusiveParts({"state_sensor"})
        }

        # hack to get this working until platforms are fixed
        self.config: GymPlatformValidator = self.get_validator(**kwargs)
        self._platform = self.config.platform

        super().__init__(**kwargs)

        if isinstance(self.action_space, gym.spaces.Discrete):
            self._last_applied_action = 0
        elif isinstance(self.action_space, gym.spaces.Box):
            self._last_applied_action = self.action_space.low

        self._operable = True

    @property
    def get_validator(self) -> typing.Type[GymPlatformValidator]:
        return GymPlatformValidator

    @property
    def observation_space(self):
        """
        Provides the observation space for a sensor to use

        Returns:
            gym.Space -- the observation space of the platform gym environment
        """
        return self._platform.observation_space

    @property
    def action_space(self):
        """
        Provides the action space for a controller to use

        Returns:
            gym.Space -- the action space of the platform gym environment
        """
        return self._platform.action_space

    def get_applied_action(self):
        """returns the action stored in this platform

        Returns:
            typing.Any -- any sort of stored action
        """
        return self._last_applied_action

    def save_action_to_platform(self, action):
        """
        saves an action to the platform if it matches
        the action space

        Arguments:
            action typing.Any -- The action to store in the platform

        Raises:
            RuntimeError: if the action attempted to be stored does not match
                        the environments action space
        """
        if not self.action_space.contains(action):
            raise RuntimeError("Error: action attempting to be stored in platform does not match platforms action space")
        self._last_applied_action = action

    @property
    def operable(self):
        return self._operable

    @operable.setter
    def operable(self, value):
        self._operable = value


class OpenAiGymInclusivePartsPlatform(OpenAiGymPlatform):
    """
    The OpenAiGymInclusivePartsPlatform mirrors OpenAiGymPlatform but without
    mutually exclusive parts
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.action_space, gym.spaces.Discrete):
            self._last_applied_action = 0
        elif isinstance(self.action_space, gym.spaces.Box):
            self._last_applied_action = self.action_space.low

        self._operable = True


class OpenAIGymSimulatorValidator(BaseSimulatorValidator):  # pylint: disable=too-few-public-methods
    """
    Validator for OpenAIGymSimulatorValidator

    gym_env: the name of a gym environment registered to the gym
            registry
    """
    # todo: maybe switch this to a PyObject and do a validator that it
    # implements gym.core.Env
    gym_env: str
    gym_configs: typing.Mapping[str, typing.Optional[typing.Union[bool, float, int, str]]] = {}
    seed: int = 1
    agent_configs: typing.Mapping[str, GymAgentConfig]
    wrappers: typing.List[PyObject] = []


class OpenAIGymSimulator(BaseSimulator):
    """
    Simulator backend for running openai Gyms
    """

    @property
    def get_simulator_validator(self) -> typing.Type[OpenAIGymSimulatorValidator]:
        """Return validator"""
        return OpenAIGymSimulatorValidator

    def __init__(self, **kwargs) -> None:
        self.config: OpenAIGymSimulatorValidator
        super().__init__(**kwargs)
        self._state = StateDict()
        self.gym_env_dict = {}
        for agent_name in self.config.agent_configs:
            env = gym.make(self.config.gym_env, **self.config.gym_configs)
            for wrapper_cls in self.config.wrappers:
                env = wrapper_cls(env)
            self.gym_env_dict[agent_name] = env
            self.gym_env_dict[agent_name].seed(self.config.seed)
        self.sim_platforms: typing.List = []
        self._time = 0.0

    def get_platforms(self):
        """
        gets the current state of the simulation and makes the sim platforms

        Returns:
            typing.List[OpenAiGymPlatform] -- the list of openai gym platforms
        """
        sim_platforms = []
        for agent_name, agent_env in self.gym_env_dict.items():
            sim_platforms.append(
                self.config.agent_configs[agent_name].platform_config.platform_class(
                    platform_name=agent_name,
                    platform=agent_env,
                    parts_list=self.config.agent_configs[agent_name].parts_list,
                    disable_exclusivity_check=self.config.disable_exclusivity_check,
                )
            )
        return sim_platforms

    def update_sensor_measurements(self):
        """
        Update and caches all the measurements of all the sensors on each platform
        """
        for plat in self.sim_platforms:
            for sensor in plat.sensors:
                sensor.calculate_and_cache_measurement(state=self._state)

    def reset(self, config):
        self._time = 0.0
        self._state.clear()
        self._state.obs = {}
        self._state.rewards = {}
        self._state.dones = {}
        self._state.info = {}
        for agent_name, agent_env in self.gym_env_dict.items():
            self._state.obs[agent_name] = agent_env.reset()

        self.sim_platforms = self.get_platforms()
        self.update_sensor_measurements()
        return self._state

    def step(self):
        for sim_platform in self.sim_platforms:
            agent_name = sim_platform.name
            if sim_platform.operable:
                tmp = self.gym_env_dict[agent_name].step(sim_platform.get_applied_action())
                self._state.obs[agent_name] = tmp[0]
                self._state.rewards[agent_name] = tmp[1]
                self._state.dones[agent_name] = tmp[2]
                self._state.info[agent_name] = tmp[3]
                if self._state.dones[agent_name]:
                    sim_platform.operable = False

        self.update_sensor_measurements()
        self._time += 1
        return self._state

    @property
    def sim_time(self) -> float:
        return self._time

    @property
    def platforms(self) -> typing.List:
        return self.sim_platforms

    def mark_episode_done(self, done_info, episode_state):
        pass

    def save_episode_information(self, dones, rewards, observations):
        pass

    def render(self, state, mode="human"):  # pylint: disable=unused-argument
        """only render first environment
        """
        agent = self.gym_env_dict.keys()[0]
        self.gym_env_dict[agent].render(mode)


PluginLibrary.AddClassToGroup(OpenAIGymSimulator, "OpenAIGymSimulator", {})

if __name__ == "__main__":

    tmp_config = {
        "gym_env": "CartPole-v1",
        "agent_configs": {
            "blue0": {
                # the platform config object needs to be a Basecontroller or BaseSensor, this is just
                # to make pydantic happy
                "sim_config": {},
                "platform_config": [
                    ("corl.simulators.openai_gym.gym_sensors.GymStateSensor", {}),
                    ("corl.simulators.openai_gym.gym_controllers.OpenAIGymMainController", {})
                ]
            }
        }
    }

    tmp_sim = OpenAIGymSimulator(**tmp_config)
    sim_state = tmp_sim.reset(None)
    print(tmp_sim.platforms[0].name, tmp_sim.platforms[0].action_space)
    print(tmp_sim.platforms[0].observation_space)
    step_result = tmp_sim.step()
    print(tmp_sim.platforms[0].sensors[0].get_measurement())
