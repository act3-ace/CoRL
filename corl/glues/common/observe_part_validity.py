"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from collections import OrderedDict
from functools import cached_property

from corl.glues.base_glue import BaseAgentPlatformGlue, BaseAgentPlatformGlueValidator
from corl.libraries.property import DictProp, DiscreteProp
from corl.libraries.units import corl_quantity
from corl.simulators.base_parts import BasePlatformPart
from corl.simulators.common_platform_utils import get_part_by_name


class ObservePartValidityValidator(BaseAgentPlatformGlueValidator):
    """
    part: which part to find on the platform
    """

    part: str


class ObservePartValidity(BaseAgentPlatformGlue):
    """Glue to observe a part's validity flag"""

    class Fields:
        """
        Fields in this glue
        """

        VALIDITY_OBSERVATION = "validity_observation"

    @staticmethod
    def get_validator() -> type[ObservePartValidityValidator]:
        return ObservePartValidityValidator

    def __init__(self, **kwargs) -> None:
        self.config: ObservePartValidityValidator
        super().__init__(**kwargs)

        self._part: BasePlatformPart = get_part_by_name(self._platform, self.config.part)
        self._part_name: str = self.config.part

        self._uname = f"ObserveValidity_{self._part_name}"

    def get_unique_name(self) -> str:
        """Class method that retrieves the unique name for the glue instance"""
        return f"ObserveValidity_{self._part_name}"

    @cached_property
    def observation_prop(self):
        return DictProp(spaces={self.Fields.VALIDITY_OBSERVATION: DiscreteProp(n=2)})

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units):
        """Observation Values"""
        d = OrderedDict()
        d[self.Fields.VALIDITY_OBSERVATION] = corl_quantity()(int(self._part.valid), "dimensionless")
        return d
