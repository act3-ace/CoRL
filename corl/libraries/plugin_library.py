"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------

PluginLibrary.AddClassToGroup(CLASS, "NAME", {"simulator": INTEGRATION_CLASS, "platform_type": AvailablePlatformTypes.XXXX})
"""
import importlib
import inspect
import itertools
import pkgutil
import sys
import typing
from traceback import print_tb

import numpy as np


class _PluginLibrary:
    """Class defining the plugin Library"""

    def __init__(self) -> None:
        self._groups: dict[str, list[tuple[typing.Callable, dict]]] = {}
        self.__PLATFORM_TYPES_LOC = "__PLATFORM_TYPES__"

    @staticmethod
    def add_paths(plugin_packages: list[str]):
        """
        loops through a list of strings (which are strings python paths)
        then recursively walks through subdirectories of those paths
        and imports them, which will cause any side effect of importing them
        such as adding a class to the plugin library
        """

        def pkg_error(module_name):
            print(f"Error in walk_packages module {module_name}")
            _, _, traceback = sys.exc_info()
            print_tb(traceback)
            raise ImportError

        for root_pkg in plugin_packages:
            root_import = importlib.import_module(root_pkg)
            for module in pkgutil.walk_packages(root_import.__path__, f"{root_import.__name__}.", onerror=pkg_error):
                importlib.import_module(module.name)

    def AddClassToGroup(self, regclass: typing.Callable, group_name: str, conditions: dict[str, list[typing.Any] | typing.Any]):
        """Add a type (or set of types) to a group name to be invoked with the provided conditions"""

        # TODO: delete this? I think? DO NOT MERGE CORL3 with out deleting or uncommenting
        # if there exists an entry with given group_name and conditions throw a RuntimeError
        # if group_name in self._groups:
        #     for group_to_check in self._groups[group_name]:
        #         if group_to_check[1] == conditions:
        #             raise RuntimeError(f"An instance with provided conditions has already been added: {group_name} -> {conditions}")

        conditions_list = []  # type: ignore
        if not isinstance(conditions, dict):
            raise RuntimeError(
                f"The conditions provided to register the class {regclass.__name__} "
                f"to group {group_name} were not a dictionary, were {type(conditions)}"
            )
        # if a user provides a list to a condition, this means they want to register more than one condition
        if any(isinstance(x, list) for x in conditions.values()):
            tmp_conditions = [cond if isinstance(cond, list) else [cond] for cond in conditions.values()]
            product_result = itertools.product(*tmp_conditions)

            conditions_list.extend(dict(zip(conditions.keys(), product_tuple)) for product_tuple in product_result)
        else:
            conditions_list.append(conditions)

        if not callable(regclass):
            raise RuntimeError(f"The {regclass} being registered to {group_name} is expected to be a callable. but it is {type(regclass)}")
        if inspect.isabstract(regclass):
            raise RuntimeError(
                f"The {regclass.__name__} being registered to {group_name} "
                f"is expected to be concrete. See MRO {inspect.getmro(type(regclass))}"
            )

        if group_name not in self._groups:
            self._groups[group_name] = []

        # TODO make sure the keys to provided conditions match with entries already in group
        for condition in conditions_list:
            self._groups[group_name].append((regclass, condition))

    def GroupExists(self, group_name: str) -> bool:
        """Determine if provided group name exists"""
        return group_name in self._groups

    def GroupMembers(self, group_name: str) -> list[tuple[typing.Callable, dict]]:
        """Return the members of the given group"""
        return self._groups[group_name]

    def FindGroup(self, reg_class: typing.Callable):
        """Return the group a class belongs to"""
        for group_name, group_list in self._groups.items():
            for item_tuple in group_list:
                if reg_class in item_tuple:
                    return group_name
        raise RuntimeError(f"Class {reg_class} not found in PluginLibrary")

    def FindMatch(self, group_name: str, condition: dict) -> typing.Callable:
        """Given a group and conditions find the associated list of types that match

        Returns None if the given group does not exists
        Raises an exception if the group exists but no match to provided could be identified
        """

        # first check that group_name exists in _group
        if group_name not in self._groups:
            raise RuntimeError(f"No items were found to be registered to group: {group_name}")

        tuple_items = self._groups[group_name]

        # in the case of no conditions
        if len(condition) == 0:
            if len(tuple_items) > 1:
                raise RuntimeError(f"In the group {group_name}, with no conditions, more than one match was established")
            return tuple_items[0][0]

        # otherwise determine the best match, or even if there is a match
        mapping_results = [difference_metric(condition, reg_tuple[1]) for reg_tuple in tuple_items]

        if np.allclose(mapping_results, 0):
            raise RuntimeError(f"In the group {group_name}, an instance matching the given conditions {condition} could not be established")

        max_index = np.argmax(mapping_results)

        return tuple_items[max_index][0]

    def add_platform_to_sim(self, platform_class: typing.Callable, sim_class: typing.Callable) -> None:
        """
        Utility for adding a platform mapping to a simulator
        """
        PluginLibrary.AddClassToGroup(platform_class, self.__PLATFORM_TYPES_LOC, {"platform": platform_class, "simulator": sim_class})

    def get_platform_from_sim_and_config(self, sim_class: typing.Callable, config: dict[str, typing.Any]) -> typing.Callable:
        """
        Utility for getting a platform type from the sim class and platform config dict
        """
        found_platforms = [
            env_plt[0]
            for env_plt in PluginLibrary.GroupMembers(self.__PLATFORM_TYPES_LOC)
            if env_plt[1]["simulator"] == sim_class and env_plt[0].match_model(config)  # type: ignore
        ]

        if not found_platforms:
            found_platforms = [
                env_plt[0] for env_plt in PluginLibrary.GroupMembers(self.__PLATFORM_TYPES_LOC) if env_plt[1]["simulator"] == sim_class
            ]
            raise RuntimeError(
                f"Unable to find a matching platform for the target configuration\n{config}"
                f"platforms classes registered to sim but not matching were: {found_platforms}"
            )

        if len(found_platforms) > 1:
            raise RuntimeError(
                f"Found {len(found_platforms)} matching platform for the target configuration\n{config}\n"
                f"platform classes found were: {found_platforms}"
            )

        return found_platforms[0]


def difference_metric(x1: dict[str, typing.Any], x2: dict[str, typing.Any]):
    """
    Heuristic for determining how much of a match 2 dictionaries are
    Dictionaries may only be a single layer or this may not work correctly

    This heuristic is calculated as follows

    number of conditions matches /
    (total number of keys in both dicts /2)

    this allows any match to get selected, and prioritize
    dictionaries with more matches

    in the event that x2 is empty the match score is set to
    0.5 / (total number of keys in both dicts /2). This essentially
    is a 0.5 match of a key and allows dictionaries with a single
    match to have a higher score while empty dicts to result in a
    non-zero match. Specifically, this is used for plugins with
    no conditions.

    Furthermore, all keys that exist in both x1 and x2 must match or a
    metric of 0 is returned.

    Arguments:
        x1 {dict} -- The dictionary used to query the list of dictionaries
        x2 {dict} -- The dictionary being compared to x1

    Returns:
        float -- The heuristic value of the comparison between the two
                 1 is max - every single field matched)
                 0 is minimum - no field matches)
                 0 is minimum - field present in both dictionaries does not match
    """

    match_score = 0
    total = (len(x1) + len(x2)) / 2

    if not x2:
        return 0.5 / total

    for key, value in x2.items():
        if key not in x1:
            continue
        if value == x1[key]:
            match_score += 1
        else:
            return 0
    return match_score / total


PluginLibrary = _PluginLibrary()
