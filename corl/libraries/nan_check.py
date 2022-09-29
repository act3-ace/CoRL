"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

NaN check module
"""
import traceback

import numpy as np


def print_trace():
    """
    The following function adds the trace back for the `NaN` check below.
    Ensures that the user knows which calling function caused the NaN
    - Typical that NaN is caused by damaged/broken platform
    """
    for line in traceback.format_stack():
        print(line.strip())


def nan_check_result(data, skip_trace=False):
    """
    Checks for nan in np array
    """
    if np.isscalar(data):
        if np.isnan(data):
            if not skip_trace:
                print_trace()
            raise ValueError("Data contains nan")
    # special case for repeated space
    elif isinstance(data, dict):
        list(map(nan_check_result, data.values()))
    elif isinstance(data, list):
        list(map(nan_check_result, data))
    elif data is None:
        if not skip_trace:
            print_trace()
        raise ValueError("Data contains nan/None")
    elif data.shape == ():
        if np.isnan(data):
            if not skip_trace:
                print_trace()
            raise ValueError("Data contains nan")
    elif len(data.shape) == 1:
        if any(np.isnan(data)):
            if not skip_trace:
                print_trace()
            raise ValueError("Data contains nan")
    else:
        if np.isnan(data).any():
            if not skip_trace:
                print_trace()
            raise ValueError("Data contains nan")
    return data
