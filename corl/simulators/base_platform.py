"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Base Platform Abstract Class Object
"""
import abc
import typing

from pydantic import BaseModel

from corl.simulators.base_parts import BaseController, BaseSensor, MutuallyExclusiveParts


class BasePlatformValidator(BaseModel):
    """BasePlatformValidator

    Parameters
    ----------
        platform_name: str
            name of platform
        parts_list: typing.List[typing.Tuple]
            list of parts the agent uses to interact with the platform
        exclusive_part_dict: typing.Optional[typing.Dict] = None
            list of mutually exlusive parts for platform
        disable_exclusivity_check: bool = False
            bool determine if part exclusivity should be check
    """
    platform_name: str
    parts_list: typing.List[typing.Tuple]
    exclusive_part_dict: typing.Optional[typing.Dict] = None
    disable_exclusivity_check: bool = False

    class Config:
        """Allow arbitrary types for Parameter"""
        arbitrary_types_allowed = True


class BasePlatform(abc.ABC):
    """
    BasePlatform base abstraction for a platform object. Platforms can be aircraft, land vehicles,
    ground radar, satellites etc.
    Platforms have platform properties that describe the platform, for example position, velocity, etc.
    Platforms have platform parts which consist of controls, or sensors.
    """

    def __init__(self, **kwargs):

        self.config: BasePlatformValidator = self.get_validator(**kwargs)

        # set default parts and mutually exclusive parts
        if self.config.exclusive_part_dict is None or self.config.disable_exclusivity_check:
            self.config.exclusive_part_dict = {
                BaseController: MutuallyExclusiveParts(set(), allow_other_keys=self.config.disable_exclusivity_check),
                BaseSensor: MutuallyExclusiveParts(set(), allow_other_keys=self.config.disable_exclusivity_check),
            }

        self._exclusive_parts = self.config.exclusive_part_dict
        #        self._platform = plaform
        self._name = self.config.platform_name
        self._sensors = self._get_part_list(self.config.parts_list, BaseSensor)
        self._controllers = self._get_part_list(self.config.parts_list, BaseController)

        self.verify_unique_parts()

    @property
    def get_validator(self) -> typing.Type[BasePlatformValidator]:
        """
        get validator for this BasePlatform

        Returns:
            BasePlatformValidator -- validator the platform will use to generate a configuration
        """
        return BasePlatformValidator

    def _get_part_list(self, part_class_list, part_base_class):
        """
        Get a list of the platform parts after being wrapped in the base integration classes.
        This will do the mapping from an SimSensor to a type of BaseSensor for example.
        Some parts have no matching sim counterpart and will not have None in the map instead
        of the part name

        Parameters
        ----------
        part_class_map: mapping of the part name to the pairs of base classes and config for that part
        parts: the parts we will wrap with the above mapping

        Returns
        -------
        A list of the wrapped platform parts
        """
        part_list = [
            part_class(self, part_config) for part_class, part_config in part_class_list if issubclass(part_class, part_base_class)
        ]

        if not self._exclusive_parts[part_base_class].are_platform_parts_mutually_exclusive(*part_list):
            lister = self._exclusive_parts[part_base_class].get_duplicate_parts(*part_list)
            fail_string = "\n".join([str(temp_var) for temp_var in lister])
            raise ValueError(f"Tried to use mutually exclusive controller platform parts:\n{fail_string} ")

        return part_list

    def verify_unique_parts(self):
        """
        Verify all parts have a unique name
        """
        for parts in [self._sensors, self._controllers]:
            part_names = list()
            for part in parts:
                if part.name in part_names:
                    raise RuntimeError("The " + part.name + " part has a unique name, but it already exists")
                part_names.append(part.name)

    @property
    def name(self) -> str:
        """
        name the name of this object

        Returns
        -------
        str
            The name of this object
        """
        return self._name

    @property
    @abc.abstractmethod
    def operable(self) -> bool:
        """Is the platform operable?

        Returns
        -------
        bool
            Is the platform operable?
        """

    @property
    def sensors(self):
        """
        Sensors attached to this platform

        Returns
        ------
        List
            list of all sensors attached to this platform
        """
        return self._sensors

    @property
    def controllers(self):
        """
        Controllers attached to this platform

        Returns
        ------
        List
            list of all controllers attached to this platform
        """
        return self._controllers
