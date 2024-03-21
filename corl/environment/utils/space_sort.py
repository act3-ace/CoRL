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

import gymnasium


def gymnasium_space_sort(space: gymnasium.spaces.Space):
    """
    space:
        Any gymnasium.spaces.Space (or nested hierarchy of spaces) to be sorted. This sorts
        any gymnasium.space.Dict in the hierarchy in KEY order.
        This is necessary as a workaround to a bug in ray/rllib which causes the action space that
        is generated from a dictionary to be
        in KEY order (the result of this line:
        https://github.com/ray-project/ray/blob/9c26d6c6de3016eb7409db8eca51f8cac4b2944e/rllib/utils/exploration/random.py#L119
        which uses https://tree.readthedocs.io/en/latest/api.html#tree.map_structure, which internally uses
        https://tree.readthedocs.io/en/latest/api.html#tree.flatten which, per the docs, causes the returned map to be in KEY order).
    """
    if isinstance(space, gymnasium.spaces.Dict):
        space_dict: gymnasium.spaces.Dict = space
        for key in space_dict.spaces:
            sub_space = space_dict.spaces[key]
            gymnasium_space_sort(sub_space)
        space.spaces = OrderedDict(sorted(space_dict.spaces.items()))
    if isinstance(space, gymnasium.spaces.Tuple):
        space_tuple: gymnasium.spaces.Tuple = space
        for sub_space in space_tuple.spaces:
            gymnasium_space_sort(sub_space)
