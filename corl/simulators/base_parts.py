"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Integration module provides abstraction for integration to enable RL environment connect to
simulation environments and/or real environments. The concept is that if we build either
simulation or real environments from this common interface we can transition between them.

- For example if there is an Sim1StickController and a Sim2StickController the interface
between them is derived from BaseStickController and thus to the policy the interface will
be the same.

The base classes have properties which determine what the deriving classes must
adhere to in order to comply with the intention of the base class.

- For example the BasePlatform has a position property which has a position_properties
(a MultiBoxProp object) that determines the ranges and types of the position property for
the BasePlatform.

To differentiate from the agent properties, base integration nomenclature uses control and
measurements for the base parts, in contrast to the nomenclature of the agent which is usually
action and observation. In this sense actions are made up of controls, and observations are
made up of measurements.
"""
from __future__ import annotations

import abc
import numbers
import typing
import warnings
from operator import attrgetter

import numpy as np
from pydantic import BaseModel, PyObject, validator

import corl.simulators.base_properties as base_props
from corl.libraries.nan_check import nan_check_result
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import Prop


class MutuallyExclusiveParts():
    """
    Class to check to see/controll mutually exclusive parts
    """

    def __init__(self, exclusive_parts, allow_other_keys=False):
        """
        exclusive_parts: The parts to check exclusivity
        allow_other_keys: allows exclusivity other than the parts defined here
        """
        self._exclusive_parts = exclusive_parts
        self.allow_other_keys = allow_other_keys

    def are_platform_parts_mutually_exclusive(self, *args: BasePlatformPart):
        """Checks to see if platform parts are mutually exclusive

        Returns:
            [type] -- [description]
        """
        parts = [platform_part.exclusiveness for platform_part in args]
        return self.are_parts_mutually_exclusive(*parts)

    def are_parts_mutually_exclusive(self, *args):
        """[summary]

        Returns:
            [type] -- [description]
        """
        total_set = set()
        for part_exclusive_set in args:
            for part_exclusiveness in part_exclusive_set:
                if part_exclusiveness in total_set:
                    return False
                if part_exclusiveness in self._exclusive_parts:
                    total_set.add(part_exclusiveness)
                elif not self.allow_other_keys:
                    raise RuntimeError(
                        f"Error: you attempted to add a part with the exclusivness {part_exclusiveness}, but "
                        f"this plaforms exclusive parts for this component type were {self._exclusive_parts}, "
                        "and the platform specified to not allow other exclusivity"
                    )
        return True

    def get_duplicate_parts(self, *args):
        """[summary]

        Returns:
            [type] -- [description]
        """
        total_list = []
        for mutually_exclusive_part in args:
            total_list += list(mutually_exclusive_part.exclusiveness)
        ret_list = []
        for exclusive_key in self._exclusive_parts:
            if total_list.count(exclusive_key) > 1:
                ret_list.append(exclusive_key)
        return ret_list


class BasePlatformPartValidator(BaseModel):
    """
    name: the optional name for this part, the class name
    will be used otherwise
    """
    part_class: PyObject
    name: typing.Optional[str] = None
    initial_validity: bool = True
    properties: typing.Optional[typing.Dict] = dict()

    @validator('name', always=True)
    def check_name(cls, v, values):
        """Check if agent subclass AgentBase"""
        if v is None:
            assert 'part_class' in values
            v = PluginLibrary.FindGroup(values['part_class'])
        return v


class BasePlatformPart(abc.ABC):
    """
    BasePlatformPart abstract class for the classes that will be part of the BasePlatform.
    This includes controls,and sensors
    """

    def __init__(self, parent_platform, config, property_class) -> None:
        config["part_class"] = self.__class__
        self.config = self.get_validator(**config)
        self._properties = property_class(**self.config.properties)
        self._parent_platform = parent_platform
        self._valid = self.config.initial_validity

    @property
    def valid(self) -> bool:
        """
        Signifies if measurements from this part should be trusted as having
        data that is valid or important to training

        Returns
        -------
        bool
            The current state of the part being valid or not
        """
        return self._valid

    def set_valid(self) -> None:
        """
        Signifies that this part should transition to being valid
        """
        self._valid = True

    def set_invalid(self) -> None:
        """
        Signifies that this part should transition to being invalid
        """
        self._valid = False

    @property
    def name(self) -> typing.Optional[str]:
        """
        The name for this platform part

        Returns
        -------
        str
            The name for this platform part
        """
        return self.config.name

    @property
    def parent_platform(self) -> 'BasePlatform':  # type: ignore  # noqa: F821
        """
        The parent platform this platform part is attached to

        Returns
        -------
        BasePlatform
            The parent platform this platform part is attached to
        """
        return self._parent_platform

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return set()

    @property
    def get_validator(self) -> typing.Type[BasePlatformPartValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return BasePlatformPartValidator

    @classmethod
    def embed_properties(cls, property_class: typing.Type[Prop]) -> typing.Type[BasePlatformPart]:
        """Embed the properties in the class definition."""

        class DynamicPlatformPart(cls):  # type: ignore[valid-type,misc]  # pylint: disable=missing-class-docstring

            def __init__(self, parent_platform, config) -> None:
                super().__init__(parent_platform=parent_platform, config=config, property_class=property_class)

        DynamicPlatformPart.__doc__ = cls.__doc__
        DynamicPlatformPart.__name__ += f':{cls.__name__}:{property_class.__name__}'
        DynamicPlatformPart.__qualname__ += f':{cls.__name__}:{property_class.__name__}'
        DynamicPlatformPart.embedded_properties = property_class

        return DynamicPlatformPart


class BaseController(BasePlatformPart, abc.ABC):
    """
    BaseController base abstraction for a controller. A controller is used to move a platform with action commands.
    The actions are usually changing the desired rates or applied forces to the platform.
    """

    @property
    def control_properties(self) -> Prop:
        """
        The properties of the control given to the apply_control function

        Returns
        -------
        Prop
            The properties of the control given tot he apply_control function
        """
        return self._properties

    def validate_control(self, control: np.ndarray) -> None:
        """
        The generic method to validate a control for this controller.

        Parameters
        ----------
        control
            The control to be validated
        """
        if not self.control_properties.create_space().contains(control):
            raise ValueError(f"{type(self).__name__} control {control} not in space {self.control_properties.create_space()} values")

    @abc.abstractmethod
    def apply_control(self, control: np.ndarray) -> None:
        """
        The generic method to apply the control for this controller.

        Parameters
        ----------
        control
            The control to be executed by the controller
        """
        ...

    @abc.abstractmethod
    def get_applied_control(self) -> typing.Union[np.ndarray, numbers.Number]:
        """
        Get the previously applied control that was given to the apply_control function
        Returns
        -------
        previously applied control that was given to the apply_control function
        """

    def get_validated_applied_control(self) -> typing.Union[np.ndarray, numbers.Number]:
        """
        Get the previously applied control with nan check
        Returns
        -------
        previously applied control that was given to the apply_control function
        """
        return nan_check_result(self.get_applied_control())


class BaseSensor(BasePlatformPart, abc.ABC):
    """
    BaseSensor base abstraction for a sensor. A sensor is a attached to a platform
    and provides information about the environment.
    """

    def __init__(self, parent_platform, config, property_class) -> None:
        super().__init__(parent_platform=parent_platform, config=config, property_class=property_class)
        self._last_measurement: typing.Optional[typing.Union[np.ndarray, typing.Tuple, typing.Dict]] = None

    @property
    def measurement_properties(self) -> Prop:
        """
        The properties of the object returned by the get_measurement function

        Returns
        -------
        Prop
            The properties of the measurement returned by the get_measurement function
        """
        return self._properties

    @abc.abstractmethod
    def _calculate_measurement(self, state: typing.Tuple) -> typing.Union[np.ndarray, typing.Tuple, typing.Dict]:
        """
        The generic method to get calculate measurements from this sensor. This is used to calculate
         the measurement to be returned by
        the get_measurement function. This allows caching the measurement to avoid re-calculation

        Parameters
        ----------
        state: typing.Tuple
            The current state of the environment used to obtain the measurement

        Returns
        -------
        typing.Union[np.ndarray, typing.Tuple, typing.Dict]
            The measurements from this sensor
        """
        ...

    def calculate_and_cache_measurement(self, state: typing.Tuple):
        """
        Calculates the measurement and caches the result in the _last_measurement variable

        Parameters
        ----------
        state: typing.Tuple
            The current state of the environment used to obtain the measurement
        """
        measurement = self._calculate_measurement(state)
        try:
            nan_check_result(measurement, True)
            self._last_measurement = measurement
        except ValueError:
            warnings.warn(
                "The mover has produced a state that is invalid - NaNs --- Code is going to set broken/damaged - Reuse last state"
            )
            # raise ValueError(f"Error calculating Measurement in {self.__class__}\n" f"Measurement: {self._last_measurement}\n") from err
        if self._last_measurement is None:
            raise ValueError('Measurement is None')

    def get_measurement(self) -> typing.Union[np.ndarray, typing.Tuple, typing.Dict, typing.List]:
        """
        The generic method to get measurements from this sensor.

        Returns
        -------
        typing.Union[np.ndarray, typing.Tuple, typing.Dict]
            The measurements from this sensor
        """
        if self._last_measurement is None:
            raise ValueError(f'Measurement is None - may also want to check operable states - ({type(self)})')
        return self._last_measurement


class CommandWithArgs(BaseModel):  # type: ignore[no-redef]
    """Model for a command with its argumets."""
    command: str
    args: typing.List[typing.Any] = []
    kwargs: typing.Dict[str, typing.Any] = {}


class NoOpControllerValidator(BasePlatformPartValidator):
    """Validator for NoOpController

    platform_init_commands: Allow the user to specify platform initialization commands.  The primary purpose of this is to allow
    verification testing between different simulators under different initial conditions without the complexity of actions being
    sent to those simulators.

    Dot notation is supported, so `foo.bar.baz` will call the method `self.parent_platform.foo.bar.baz()`.

    """
    platform_init_commands: typing.List[CommandWithArgs] = []

    @validator('platform_init_commands', pre=True, each_item=True)
    def expand_command(cls, v):
        """Convert simple string form to full form with arguments."""
        if isinstance(v, str):
            return {'command': v}
        return v


class NoOpController(BaseController):
    """
    NoOpController controller define by empty prop can be use when simulator is
    creating actions for an agent.
    """

    def __init__(self, parent_platform, config, property_class) -> None:
        self.config: NoOpControllerValidator
        super().__init__(parent_platform=parent_platform, config=config, property_class=property_class)
        self.new_control = np.array([], dtype=np.float32)

        for command_data in self.config.platform_init_commands:
            func = attrgetter(command_data.command)
            func(self.parent_platform)(*command_data.args, **command_data.kwargs)

    @property
    def get_validator(self) -> typing.Type[NoOpControllerValidator]:
        """Validator for NoOpController"""
        return NoOpControllerValidator

    def apply_control(self, control: np.ndarray) -> None:
        """
        The generic method to apply the control for this controller.

        Parameters
        ----------
        control
            The control to be executed by the controller
        """
        self.new_control = control

    def get_applied_control(self):
        return self.new_control


PluginLibrary.AddClassToGroup(NoOpController.embed_properties(base_props.NoOpProp), "Controller_NoOp", {})


class BaseTimeSensor(BaseSensor):
    """Base type for a time sensor

    Arguments:
        BaseSensor -- The base class type for the sensor
    """

    def __init__(self, parent_platform, config: typing.Dict = None):
        super().__init__(parent_platform, config, base_props.TimeProp)
