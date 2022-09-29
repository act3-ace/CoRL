"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
collection utils
"""

import collections
import typing


def get_dictionary_subset(data_dict: typing.Union[typing.Dict, typing.OrderedDict], keys: typing.List) -> typing.OrderedDict:
    """
    Create dictionary of input tupe with subset of values

    Parameters
    ----------
    data_dict typing.Union[typing.Dict, typing.OrderedDict]: src dictionary
    keys list: list of keys defining subset

    Returns
    -------
    dictionary of input tupe with subset of values
    """
    return collections.OrderedDict({key: data_dict[key] for key in keys if key in data_dict})
