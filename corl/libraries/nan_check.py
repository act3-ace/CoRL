"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------

NaN check module
"""
import traceback

import numpy as np

from corl.libraries.units import Quantity


def print_trace():
    """
    The following function adds the trace back for the `NaN` check below.
    Ensures that the user knows which calling function caused the NaN
    - Typical that NaN is caused by damaged/broken platform
    """
    for line in traceback.format_stack():
        print(line.strip())


def recursive_nan_check(data, skip_trace=False):
    """
    does the nan check after quantities are extracted
    adds a stack trace for the nan checks should something fail the check
    """
    if isinstance(data, Quantity):
        recursive_nan_check(data.m, skip_trace=skip_trace)
    elif np.isscalar(data):
        if np.isnan(data):
            if not skip_trace:
                print_trace()
            raise ValueError("Data contains nan")
    # special case for repeated space
    elif isinstance(data, dict):
        list(map(recursive_nan_check, data.values()))
    elif isinstance(data, list):
        list(map(recursive_nan_check, data))
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
    elif np.isnan(data).any():
        if not skip_trace:
            print_trace()
        raise ValueError("Data contains nan")


def nan_check_result(data, skip_trace=False):
    """
    Checks for nan in np array
    """
    tmp_data = data
    if not isinstance(tmp_data, list | dict | Quantity):
        raise RuntimeError(
            "ERROR: nan_check_result was given an outer data structure that was not a "
            "Quantity, Dict, or List, sensors and controllers must always return one of these types "
            f"data struct was {data}"
        )
    recursive_nan_check(tmp_data, skip_trace=skip_trace)
    return data
