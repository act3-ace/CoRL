"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from typing import Any

import numpy as np
import pandas as pd

from corl.evaluation.connection.base_epp_update import get_updates
from corl.evaluation.runners.inference.protocol.simple_epp_update import SimpleEppUpdate
from corl.evaluation.runners.section_factories.test_cases.base_network_strategy import BaseNetworkTestCaseStrategy
from corl.libraries.units import Quantity, Unit


class SimpleNetworkEppTestCaseStrategy(BaseNetworkTestCaseStrategy[SimpleEppUpdate]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._cur_updates = SimpleEppUpdate(updates={})

    def _create_test_case_for_updates(self, updates: SimpleEppUpdate) -> pd.DataFrame | list[dict[str, Any]]:
        update_dict = get_updates(updates)

        test_case: dict[str, tuple[np.ndarray | float | int | str, Unit | str]] = {}
        value: np.ndarray | float | int | str
        units: Unit | str
        for key, qty in update_dict.items():
            if isinstance(qty, Quantity):
                value = qty.value
                units = qty.units
            else:
                value = qty
                units = "dimensionless"
            test_case[".".join(key)] = (value, units)

        return self._test_case + [test_case]  # fmt: skip  # noqa: RUF005
