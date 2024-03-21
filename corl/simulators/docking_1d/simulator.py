"""
This module defines the Docking1dSimulator class which maintains environment objects (platforms and entities)
and progresses a simulated training episode via the step() method.
"""


import numpy as np

from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_simulator import BaseSimulator, BaseSimulatorResetValidator, BaseSimulatorState, BaseSimulatorValidator
from corl.simulators.docking_1d.entities import Deputy1D
from corl.simulators.docking_1d.platform import Docking1dPlatform


class Docking1dSimulatorValidator(BaseSimulatorValidator):
    """
    A validator for the Docking1dSimulatorValidator config.

    step_size: A float representing how many simulated seconds pass each time the simulator updates
    """

    step_size: float


class Docking1dSimulatorResetValidator(BaseSimulatorResetValidator):
    """
    A Validator for Docking1dSimulatorValidator reset configs

    Parameters
    ----------
    platform_config: dict
        Contains individual initialization dicts for each agent.
        Key is platform name, value is platform's initialization dict.
    """

    platform_config: dict | None = {}


class Docking1dSimulator(BaseSimulator):
    """
    Simulator for the 1D Docking task. Interfaces Docking1dPlatform with underlying Deputy1D entities in Docking
    simulation. Docking1dSimulator is responsible for initializing the platform objects for a simulation
    and knowing how to set up episodes based on input parameters from a parameter provider.
    It is also responsible for reporting the simulation state at each timestep.
    """

    @property
    def get_simulator_validator(self):
        return Docking1dSimulatorValidator

    @property
    def get_reset_validator(self):
        return Docking1dSimulatorResetValidator

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state: BaseSimulatorState | None = None
        self.clock = 0.0

    def reset(self, config):
        config = self.get_reset_validator(**config)
        self.clock = 0.0

        # construct entities ("Gets the platform object associated with each simulation entity.")
        self.sim_entities = {}
        for agent_id in self.config.agent_configs:
            agent_reset_config = config.platforms.get(agent_id, {})
            self.sim_entities[agent_id] = Deputy1D(name=agent_id, **agent_reset_config)

        # construct platforms ("Gets the correct backend simulation entity for each agent.")
        sim_platforms = {}
        for agent_id, entity in self.sim_entities.items():
            agent_config = self.config.agent_configs[agent_id]
            sim_platforms[agent_id] = Docking1dPlatform(platform_name=agent_id, platform=entity, parts_list=agent_config.parts_list)

        self._state = BaseSimulatorState(sim_platforms=sim_platforms, sim_time=self.clock)
        self.update_sensor_measurements()
        return self._state

    def step(self, platforms_to_action):
        for agent_id, platform in self._state.sim_platforms.items():
            action = np.asarray(platform.get_applied_action().m, dtype=np.float32)
            entity = self.sim_entities[agent_id]
            entity.step(action=action, step_size=self.config.step_size)
            platform.sim_time = self.clock
        self.update_sensor_measurements()
        self.clock += self.config.step_size
        self._state.sim_time = self.clock
        return self._state

    @property
    def sim_time(self) -> float:
        return self.clock

    @property
    def platforms(self):
        return self._state.sim_platforms

    def update_sensor_measurements(self):
        """
        Update and cache all the measurements of all the sensors on each platform
        """
        for plat in self._state.sim_platforms.values():
            for sensor in plat.sensors.values():
                sensor.calculate_and_cache_measurement(state=self._state)

    def mark_episode_done(self, done_info, episode_state, metadata):
        pass

    def save_episode_information(self, dones, rewards, observations, observation_units):
        pass


# Register the Simulator with the PluginLibrary. Requires a class and a reference name.

PluginLibrary.AddClassToGroup(Docking1dSimulator, "Docking1dSimulator", {})
