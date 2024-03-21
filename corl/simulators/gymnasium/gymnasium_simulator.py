"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Base Simulator and Platform for Toy OpenAI Environments
This mainly shows a "how to use example" and provide an setup to unit test with
"""
import typing
from functools import cached_property

import gymnasium
from pydantic import BaseModel, ImportString

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.units import corl_get_ureg
from corl.simulators.base_platform import BasePlatform, BasePlatformValidator
from corl.simulators.base_simulator import AgentConfig, BaseSimulator, BaseSimulatorState, BaseSimulatorValidator


class GymnasiumPlatformValidator(BasePlatformValidator):
    """GymnasiumPlatformValidator

    Parameters
    ----------
        platform: gymnasium.Env
            Gymnasium env associated with GymnasiumPlatform
    """

    platform: gymnasium.Env


class GymnasiumPlatformConfig(BaseModel):
    """
    GymnasiumPlatformSimConfig
    """

    platform_class: ImportString


class GymnasiumAgentConfig(AgentConfig):
    """
    platform_config: any configuration needed for the simulator to
                initialize this platform and configure it in the sim class
    parts_list: a list of tuples where the first element is come python class path
                    of a BasePart, and then the second element is a configuration dictionary for that part

    Arguments:
        BaseModel: [description]
    """

    platform_config: GymnasiumPlatformConfig


class GymnasiumPlatform(BasePlatform):
    """
    The GymnasiumPlatform wraps some gymnasium environment as it's platform and
    allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function
    """

    def __init__(self, **kwargs) -> None:
        # hack to get this working until platforms are fixed
        self.config: GymnasiumPlatformValidator = self.get_validator()(**kwargs)
        self._platform = self.config.platform

        super().__init__(**kwargs)

        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            self._last_applied_action = corl_get_ureg().Quantity(0, "dimensionless")
        elif isinstance(self.action_space, gymnasium.spaces.Box):
            self._last_applied_action = corl_get_ureg().Quantity(self.action_space.low, "dimensionless")
        else:
            raise RuntimeError("shouldn't have gotten here")

        self._operable = True

    @staticmethod
    def get_validator() -> type[GymnasiumPlatformValidator]:
        return GymnasiumPlatformValidator

    @cached_property
    def observation_space(self) -> gymnasium.spaces.Space:
        """
        Provides the observation space for a sensor to use

        Returns:
            gymnasium.Space -- the observation space of the platform gymnasium environment
        """
        return self._platform.observation_space

    @cached_property
    def action_space(self) -> gymnasium.spaces.Space:
        """
        Provides the action space for a controller to use

        Returns:
            gymnasium.Space -- the action space of the platform gymnasium environment
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
        if not self.action_space.contains(action.m):
            raise RuntimeError(
                "Error: action attempting to be stored in platform does not match platforms action space\n"
                f"{action.m=} and {self.action_space=}"
            )
        self._last_applied_action = action

    @property
    def operable(self):
        return self._operable

    @operable.setter
    def operable(self, value):
        self._operable = value


class GymnasiumInclusivePartsPlatform(GymnasiumPlatform):
    """
    The GymnasiumInclusivePartsPlatform mirrors GymnasiumPlatform but without
    mutually exclusive parts
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            self._last_applied_action = 0
        elif isinstance(self.action_space, gymnasium.spaces.Box):
            self._last_applied_action = self.action_space.low

        self._operable = True

        self._last_applied_action = corl_get_ureg().Quantity(self._last_applied_action, "dimensionless")


class GymnasiumSimulatorValidator(BaseSimulatorValidator):
    """
    Validator for GymnasiumSimulatorValidator

    gymnasium_env: the name of a gymnasium environment registered to the gym
            registry
    """

    # todo: maybe switch this to a ImportString and do a validator that it
    # implements gymnasium.core.Env
    gymnasium_env: str
    gymnasium_configs: typing.Mapping[str, bool | (float | (int | str)) | None] = {}
    seed: int = 1
    agent_configs: typing.Mapping[str, GymnasiumAgentConfig]
    wrappers: list[ImportString] = []


class SimulatorState(BaseSimulatorState):
    """
    obs: the current obs for each named gymnasium env
    rewards: the current reward for each named gymnasium env
    dones: the current done for each named gymnasium env
    info: the current info for each named gymnasium env
    """

    obs: dict[str, typing.Any] = {}
    rewards: dict[str, typing.Any] = {}
    truncated: dict[str, typing.Any] = {}
    terminated: dict[str, typing.Any] = {}
    info: dict[str, typing.Any] = {}


class GymnasiumSimulator(BaseSimulator):
    """
    Simulator backend for running openai Gymnasiums
    """

    @property
    def get_simulator_validator(self) -> type[GymnasiumSimulatorValidator]:
        """Return validator"""
        return GymnasiumSimulatorValidator

    def __init__(self, **kwargs) -> None:
        self.config: GymnasiumSimulatorValidator
        super().__init__(**kwargs)
        self._state: SimulatorState
        self.gymnasium_env_dict = {}
        for agent_name in self.config.agent_configs:
            env = gymnasium.make(self.config.gymnasium_env, **self.config.gymnasium_configs)  # type: ignore
            for wrapper_cls in self.config.wrappers:
                env = wrapper_cls(env)
            self.gymnasium_env_dict[agent_name] = env
        self.sim_platforms: dict = {}
        self._time = 0.0

    def get_platforms(self):
        """
        gets the current state of the simulation and makes the sim platforms

        Returns:
            typing.Dict[str, GymnasiumPlatform] -- the dict of openai gymnasium platforms
        """
        return {
            agent_name: self.config.agent_configs[agent_name].platform_config.platform_class(
                platform_name=agent_name,
                platform=agent_env,
                parts_list=self.config.agent_configs[agent_name].parts_list,
            )
            for agent_name, agent_env in self.gymnasium_env_dict.items()
        }

    def update_sensor_measurements(self):
        """
        Update and caches all the measurements of all the sensors on each platform
        """
        for plat in self.sim_platforms.values():
            for sensor in plat.sensors.values():
                sensor.calculate_and_cache_measurement(state=self._state)

    def reset(self, config):
        self._time = 0.0
        self.sim_platforms = self.get_platforms()
        self._state = SimulatorState(sim_platforms=self.sim_platforms, sim_time=self._time)

        for agent_name, agent_env in self.gymnasium_env_dict.items():
            self._state.obs[agent_name], self._state.info[agent_name] = agent_env.reset(seed=self.config.seed)

        self.update_sensor_measurements()
        return self._state

    def step(self, platforms_to_action):
        for agent_name, sim_platform in self.sim_platforms.items():
            if sim_platform.operable:
                tmp = self.gymnasium_env_dict[agent_name].step(sim_platform.get_applied_action().m)
                self._state.obs[agent_name] = tmp[0]
                self._state.rewards[agent_name] = tmp[1]
                self._state.terminated[agent_name] = tmp[2]
                self._state.truncated[agent_name] = tmp[3]
                self._state.info[agent_name] = tmp[4]
                if self._state.terminated[agent_name] or self._state.truncated[agent_name]:
                    sim_platform.operable = False

        self.update_sensor_measurements()
        self._time += 1
        self._state.sim_time = self._time
        return self._state

    @property
    def sim_time(self) -> float:
        return self._time

    @property
    def platforms(self):
        return self.sim_platforms

    def mark_episode_done(self, done_info, episode_state, metadata):
        pass

    def save_episode_information(self, dones, rewards, observations, observation_units):
        pass

    def render(self, state, mode="human"):
        """only render first environment"""
        agent = self.gymnasium_env_dict.keys()[0]
        self.gymnasium_env_dict[agent].render(mode)


PluginLibrary.AddClassToGroup(GymnasiumSimulator, "GymnasiumSimulator", {})

if __name__ == "__main__":
    tmp_config = {
        "gymnasium_env": "CartPole-v1",
        "agent_configs": {
            "blue0": {
                # the platform config object needs to be a Basecontroller or BaseSensor, this is just
                # to make pydantic happy
                "sim_config": {},
                "platform_config": [
                    ("corl.simulators.gymnasium.gymnasium_sensors.GymnasiumStateSensor", {}),
                    ("corl.simulators.gymnasium.gymnasium_controllers.GymnasiumMainController", {}),
                ],
            }
        },
    }

    tmp_sim = GymnasiumSimulator(**tmp_config)
    sim_state = tmp_sim.reset(None)
    print(tmp_sim.platforms["blue0"].name, tmp_sim.platforms[0].action_space)
    print(tmp_sim.platforms["blue0"].observation_space)
    step_result = tmp_sim.step({"blue0"})
    print(next(iter(tmp_sim.platforms["blue0"].sensors.values())).get_measurement())
