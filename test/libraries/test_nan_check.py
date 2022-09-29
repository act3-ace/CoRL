"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import pytest
import numpy as np
from corl.libraries.nan_check import nan_check_result


def test_nan_check():

    nan_check_result(np.array([0.0, 0.0]))
    nan_check_result(np.array(0.0))
    nan_check_result(np.array(False))


    with pytest.raises(ValueError) as exec_info:
        nan_check_result(np.array(np.nan))

    with pytest.raises(ValueError) as exec_info:
        nan_check_result(np.array([np.nan,]))



