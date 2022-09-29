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
from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict

from pydantic import BaseModel, PyObject, ValidationError, parse_obj_as, validate_arguments

from corl.libraries.factory import Factory
from corl.libraries.state_dict import StateDict
from corl.libraries.units import ValueWithUnits


class AgentConfig(BaseModel):
    """
    platform_config: any configuration needed for the simulator to
                initialize this platform and configure it in the sim class
    parts_list: a list of tuples where the first element is come python class path
                    of a BasePart, and then the second element is a configuration dictionary for that part

    Arguments:
        BaseModel {[type]} -- [description]
    """
    platform_config: typing.Union[typing.Dict[str, typing.Any], BaseModel]
    parts_list: typing.List[typing.Tuple[PyObject, typing.Dict[str, typing.Any]]]


class BaseSimulatorValidator(BaseModel):
    """
    worker_index: what worker this simulator class is running on < used for render
    vector_index: what vector index this simulator class is running on < used for render
    agent_configs: the mapping of agent names to a dict describing the platform
    disable_exclusivity_check: this bool should be used to tell downstream platforms that
                                the user wishes to disable any mutually exclusive parts checking
                                on a platform

                                this should pretty much only be used for behavior tree type
                                agents
    frame_rate: the rate the simulator should run at (in Hz)
    """
    worker_index: int = 0
    vector_index: int = 0
    agent_configs: typing.Mapping[str, AgentConfig]
    disable_exclusivity_check: bool = False
    frame_rate: float = 1.0


class BaseSimulatorResetValidator(BaseModel):
    """
    Validator to use to validate the reset input to a simulator class
    allows the simulator class to take EPP params and structure/validate them

    Note that all attributes in this validator need to survive being parsed by validation_helper_units_and_parameters.

    Subclasses can redefine `platforms` to make the `typing.Any` more restrictive.  It must remain a dictionary with keys named for
    the platforms in the simulation.
    """
    platforms: typing.Dict[str, typing.Any] = {}


@validate_arguments
def validation_helper_units_and_parameters(value: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    """Recursively inspect a dictionary, converting ValueWithUnits and Factory"""

    output: typing.Dict[str, typing.Any] = {}
    for k, v in value.items():
        try:
            elem = parse_obj_as(ValueWithUnits, v)
        except ValidationError:
            pass
        else:
            output[k] = elem
            continue

        try:
            factory = parse_obj_as(Factory, v)
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


class BaseSimulator(ABC):
    """
    BaseSimulator is responsible for initializing the platform objects for a simulation
    and knowing how to setup episodes based on input parameters from a parameter provider
    it is also responsible for reporting the simulation state at each timestep
    """

    def __init__(self, **kwargs):
        self.config = self.get_simulator_validator(**kwargs)

    @property
    def get_simulator_validator(self) -> typing.Type[BaseSimulatorValidator]:
        """
        returns the validator for the configuration options to the simulator
        the kwargs to this class are validated and put into a defined struct
        potentially raising based on invalid configurations

        Returns:
            BaseSimulatorValidator -- The validator to use for this simulation class
        """
        return BaseSimulatorValidator

    @property
    def get_reset_validator(self) -> typing.Type[BaseSimulatorResetValidator]:
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

    @abstractmethod
    def reset(self, config: typing.Dict[str, typing.Any]) -> StateDict:
        """
        reset resets the simulation and sets up a new episode

        Arguments:
            config {typing.Dict[str, typing.Any]} -- The parameters to
                    validate and use to setup this episode

        Returns:
            StateDict -- The simulation state, has a .sim_platforms attr
                        to access the platforms made by the simulation
        """
        ...

    @abstractmethod
    def step(self) -> StateDict:
        """
        advances the simulation platforms and returns the state

        Returns:
            StateDict -- The state after the simulation updates, has a
                        .sim_platforms attr to access the platforms made by the simulation
        """
        ...

    @abstractproperty
    def sim_time(self) -> float:
        """
        returns the time

        Returns:
            float - time
        """
        ...

    @abstractproperty
    def platforms(self) -> typing.List:
        """
        returns a list of platforms in the simulation

        Returns:
            list of platforms
        """
        ...

    @abstractmethod
    def mark_episode_done(self, done_info: OrderedDict, episode_state: OrderedDict):
        """
        Takes in the done_info specifying how the episode completed
        and does any book keeping around ending an episode

        Arguments:
            done_info {OrderedDict} -- The Dict describing which Done conditions ended an episode
            episode_state {OrderedDict} -- The episode state at the end of the simulation
        """
        ...

    @abstractmethod
    def save_episode_information(self, dones, rewards, observations):
        """
        provides a way to save information about the current episode
        based on the environment

        Arguments:
            dones {[type]} -- the current done info of the step
            rewards {[type]} -- the reward info for this step
            observations {[type]} -- the observations for this step
        """
        ...

    def render(self, state, mode="human"):  # pylint: disable=unused-argument
        """
        allows you to do something to render your simulation
        you are responsible for checking which worker/vector index you are on
        """
        ...

    def delete_platform(self, name):  # pylint: disable=unused-argument
        """
        provides a way to delete a platform from the simulation
        """
        ...
