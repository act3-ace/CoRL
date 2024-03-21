"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from abc import abstractmethod

from corl.libraries.units import Quantity


class BaseEppUpdate:
    @abstractmethod
    def get_updates(self) -> dict[tuple[str, ...], Quantity | float | str]:
        ...


def get_updates(updates: BaseEppUpdate, prefix: str | None = None) -> dict[tuple[str, ...], Quantity | float | str]:
    modified_updates: dict[tuple[str, ...], Quantity | float | str] = {}

    key: tuple[str, ...]
    value: Quantity | float | str
    for key, value in updates.get_updates().items():
        if prefix is None:
            modified_updates[key] = value
        elif prefix and key[0] == prefix:
            modified_updates[key[1:]] = value

    return modified_updates
