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

from pydantic import BaseModel, ConfigDict

from corl.simulators.base_parts import BaseController, BaseSensor


class BasePlatformValidator(BaseModel):
    """BasePlatformValidator

    Parameters
    ----------
        platform_name: str
            name of platform
        parts_list: typing.List[typing.Tuple]
            list of parts the agent uses to interact with the platform
    """

    platform_name: str
    parts_list: list[tuple]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BasePlatform(abc.ABC):
    """
    BasePlatform base abstraction for a platform object. Platforms can be aircraft, land vehicles,
    ground radar, satellites etc.
    Platforms have platform properties that describe the platform, for example position, velocity, etc.
    Platforms have platform parts which consist of controls or sensors.
    """

    def __init__(self, **kwargs) -> None:
        self.config: BasePlatformValidator = self.get_validator()(**kwargs)

        self._name = self.config.platform_name
        self._sensors = self._get_part_dict(self.config.parts_list, BaseSensor)
        self._controllers = self._get_part_dict(self.config.parts_list, BaseController)

        self.verify_unique_parts()

    @staticmethod
    def get_validator() -> type[BasePlatformValidator]:
        """
        get validator for this BasePlatform

        Returns:
            BasePlatformValidator -- validator the platform will use to generate a configuration
        """
        return BasePlatformValidator

    def _get_part_dict(self, part_class_list, part_base_class):
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
        part_dict = {}
        for part_class, part_config in filter(lambda x: issubclass(x[0], part_base_class), part_class_list):
            tmp = part_class(self, part_config)
            if tmp.name in part_dict:
                if part_dict[tmp.name].config != tmp.config:
                    raise RuntimeError(
                        f"Platform {self._name} tried to add a part {tmp.name} to its part Dict for type {part_base_class}, "
                        f"however a part with this name already exists in the part dict with a different configuration.\n"
                        f"new part class was {part_class}, part config was {tmp.config},\n"
                        f"already existing part config is {part_dict[tmp.name].config}\n"
                        f"already existing part names are {part_dict.keys()}"
                    )
                continue
            part_dict[tmp.name] = tmp

        return part_dict

    def verify_unique_parts(self):
        """
        Verify all parts have a unique name
        """
        sensor_parts = set(self._sensors.keys())
        controller_parts = set(self._controllers.keys())
        part_union = sensor_parts.intersection(controller_parts)
        if filtered_intersection := {x for x in part_union if not isinstance(self._sensors[x], BaseController)}:
            raise RuntimeError(
                f"Parts on a platform need to have a unique name, but platform {self._name} "
                f"has the following sensors and controllers that share the same name: {filtered_intersection} "
                f"Parts may subclass both BaseSensor and BaseController and are not a concern in this check"
            )

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
