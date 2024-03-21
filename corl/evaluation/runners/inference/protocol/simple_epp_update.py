"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from pydantic import BaseModel

from corl.evaluation.connection.base_epp_update import BaseEppUpdate
from corl.libraries.units import Quantity


class SimpleEppUpdate(BaseModel, BaseEppUpdate):
    separator: str = "."

    updates: dict[str, Quantity | float | str]

    def get_updates(self) -> dict[tuple[str, ...], Quantity | float | str]:
        updates: dict[tuple[str, ...], Quantity | float | str] = {}
        for key, value in self.updates.items():
            updates[tuple(key.split(self.separator))] = value
        return updates
