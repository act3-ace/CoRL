"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import numpy as np
import pytest

from corl.libraries.nan_check import nan_check_result
from corl.libraries.units import corl_get_ureg


def test_nan_check():
    _Quantity = corl_get_ureg().Quantity

    nan_check_result(_Quantity(np.array([0.0, 0.0]), "dimensionless"))
    nan_check_result(_Quantity(np.array(0.0), "dimensionless"))
    nan_check_result(_Quantity(np.array(False), "dimensionless"))

    with pytest.raises(ValueError):
        nan_check_result(_Quantity(np.array(np.nan), "dimensionless"))

    with pytest.raises(ValueError):
        nan_check_result(
            _Quantity(
                np.array(
                    [
                        np.nan,
                    ]
                ),
                "dimensionless",
            )
        )
