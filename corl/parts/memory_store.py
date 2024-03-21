"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import abc

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp, DiscreteProp, MultiBinary
from corl.libraries.units import Quantity
from corl.parts.comms import Comms, CommsValidator
from corl.simulators.base_parts import BasePlatformPartValidator


class MemoryStoreValidator(CommsValidator):
    """Validator for memory store"""


class MemoryStore(Comms, abc.ABC):
    """Platform part for sharing intra-platform data (i.e. data between agents on the same platform)"""

    def __init__(self, parent_platform, config, property_class) -> None:
        self.config: MemoryStoreValidator
        super().__init__(parent_platform, config, property_class)

        self._data = self.default_value

    @staticmethod
    def get_validator() -> type[BasePlatformPartValidator]:
        """
        return the validator that will be used on the configuration
        of this part
        """
        return MemoryStoreValidator

    def _set_data(self, data: Quantity) -> None:
        self._data = data

    def _get_data(self) -> Quantity:
        return self._data


PluginLibrary.AddClassToGroup(MemoryStore.embed_properties(BoxProp), "BoxPropMemoryStore", {})
PluginLibrary.AddClassToGroup(MemoryStore.embed_properties(MultiBinary), "MultiBinaryPropStore", {})
PluginLibrary.AddClassToGroup(MemoryStore.embed_properties(DiscreteProp), "DiscretePropStore", {})
