"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Parse the evaluation configuration
"""
import copy
import typing
from collections import abc

# Lowest level of data read from the configuration file
LOW_DATA = int | float | str
LOW_DATA_SEQUENCE = typing.Sequence[LOW_DATA]
# All possibilities for the "value" field in the configuration file
INPUT_VALUE = LOW_DATA | LOW_DATA_SEQUENCE | typing.Mapping[str, int]

# Type of the "randomize" subtree
RANDOMIZE = typing.Mapping[str, LOW_DATA]
# Type for all subtrees under the various variable names
VARIABLE_ELEMENTS = INPUT_VALUE | RANDOMIZE

# Type for each variable within the configuration file
VARIABLE = typing.Mapping[str, VARIABLE_ELEMENTS]
# Type for the collection of all variables.  The code handles deeper recursion, such as Mapping[str, Mapping[str, INPUT_VARIABLE]].
VARIABLE_STRUCTURE = typing.Mapping[str, VARIABLE | typing.Mapping[str, VARIABLE]]

# Type for the collection of all variables after it has been flattened to remove recursion
VARIABLE_STRUCTURE_FLAT = typing.Mapping[str, VARIABLE]
MUTABLE_VARIABLE_STRUCTURE_FLAT = typing.MutableMapping[str, VARIABLE]

# Type for the collection of all variables once a single subtree is selected
MUTABLE_EXTRACTED_VARIABLE_STRUCTURE = typing.MutableMapping[str, VARIABLE_ELEMENTS]
# Type for the collection of all variables once a single subtree -- value -- is selected
EXTRACTED_VARIABLE_STRUCTURE_VALUE = typing.Mapping[str, INPUT_VALUE]
# Type for the collection of all variables once a single subtree -- randomize -- is selected
EXTRACTED_VARIABLE_STRUCTURE_RANDOMIZE = typing.Mapping[str, RANDOMIZE]

# Type for the collection of all variable values after the ranges are resolved
MUTABLE_RESOLVED_VARIABLE_STRUCTURE_VALUE = typing.MutableMapping[str, LOW_DATA_SEQUENCE]
RESOLVED_VARIABLE_STRUCTURE_VALUE = typing.Mapping[str, LOW_DATA_SEQUENCE]


def flatten_config(config: VARIABLE_STRUCTURE, _path: str | None = None) -> MUTABLE_VARIABLE_STRUCTURE_FLAT:
    """Flatten a nested configuration object to a single collection of keys.

    Parameters
    ----------
    config : VARIABLE_STRUCTURE
        Configuration tree as described by `python act3/agents/evaluation/evaluation.py --describe`.

    Returns
    -------
    MUTABLE_VARIABLE_STRUCTURE_FLAT
        Single level configuration tree.  Nested levels are replaced with expanded keys in the format "top_key.sub_key.leaf_key".

    Raises
    ------
    ValueError
        An input key contains the character "."
    TypeError
        The value is not a mapping.

    Notes
    -----
    This function is invertible by `unflatten`.
    """
    output: MUTABLE_VARIABLE_STRUCTURE_FLAT = {}

    for key, value in config.items():
        if "." in key:
            raise ValueError(f'Cannot use "." in key: {key}')
        next_path = key if _path is None else f"{_path}.{key}"

        if not isinstance(value, abc.Mapping):
            raise TypeError(f"Subtree must be a mapping at {next_path}")

        if "value" in value:
            output[next_path] = copy.deepcopy(typing.cast(VARIABLE, value))
        else:
            output.update(flatten_config(typing.cast(typing.Mapping[str, VARIABLE], value), _path=next_path))

    return output


def unflatten(obj: typing.Mapping[str, typing.Any]) -> typing.Mapping[str, typing.Any]:
    """Returned a flattened object to its nested representation

    Parameters
    ----------
    obj : Mapping[str, Any]
        Flattened representation of a nested mapping.  The format of the initial nested mapping is represented by the structure of the keys
        in the format "top_key.sub_key.leaf_key".

    Returns
    -------
    Mapping[str, Any]
        Nested representation of the input mapping.  For example `{'top_key': {'sub_key': {'leaf_key': value}}}`.

    Notes
    -----
    If all values are mappings with the key "value", then this function is invertible by `flatten_config`.
    """
    output: typing.MutableMapping[str, typing.Any] = {}

    for key, value in obj.items():
        key_split = key.split(".")
        suboutput = output
        for subkey in key_split[:-1]:
            if subkey not in suboutput:
                suboutput[subkey] = {}
            suboutput = typing.cast(typing.MutableMapping[str, typing.Any], suboutput[subkey])
        suboutput[key_split[-1]] = value

    return output


def extract_config_element(config: VARIABLE_STRUCTURE_FLAT, element: str = "value") -> MUTABLE_EXTRACTED_VARIABLE_STRUCTURE:
    """Extract the items of a particular element from the dictionary

    Parameters
    ----------
    config : VARIABLE_STRUCTURE_FLAT
        As produced by flatten_config
    element : str, optional
        The element to extract from each dictionary.  The default is "value".

    Returns
    -------
    MUTABLE_EXTRACTED_VARIABLE_STRUCTURE
        Configuration information in the same structure as the input; however, each variable is reduced to only contain the data from its
        value key.

    Examples
    --------
    Input:
    config = {'name': {'value': [0], 'something': 1},
              'group.subkey': {'value': {'min': 1, 'max': 10, 'step': 1}, 'another_key': 3},
              'hello': {'who': 'world'}
              }
    element = 'value'

    Output:
    {'name': [0], 'group.subkey': {'min': 1, 'max': 10, 'step': 1}}

    Raises
    ------
    TypeError
        Values are not mappings.
    """
    output: MUTABLE_EXTRACTED_VARIABLE_STRUCTURE = {}

    for key, data in config.items():
        if not isinstance(data, abc.Mapping):
            raise TypeError(f"Data for {key} is not a mapping")

        if element in data:
            output[key] = data[element]

    return output


def resolve_config_ranges(config: EXTRACTED_VARIABLE_STRUCTURE_VALUE) -> MUTABLE_RESOLVED_VARIABLE_STRUCTURE_VALUE:
    """Resolve single values and min/max/step ranges into explicitly listed value arrays

    Parameters
    ----------
    config : EXTRACTED_VARIABLE_STRUCTURE_VALUE
        Configuration tree as if produced by `extract_config_element` with element `value`.

    Returns
    -------
    MUTABLE_RESOLVED_VARIABLE_STRUCTURE_VALUE
        Configuration tree with the same keys as the input; however, the values are all arrays.  Single values become one-element arrays.
        Min/max/step ranges are expanded into their explicitly listed values.

    Raises
    ------
    ValueError
        Maximum is less than minimum
        Unsupported structure, likely caused by missing "min", "max", or "step" keys, or extra keys provided
    TypeError
        Min/max/step values are not integers
    """

    output: MUTABLE_RESOLVED_VARIABLE_STRUCTURE_VALUE = {}

    for key, data in config.items():
        if isinstance(data, abc.Mapping):
            if set(data.keys()) != {"min", "max", "step"}:
                raise ValueError(f"Unsupported structure at {key}")

            if not all(isinstance(v, int) for v in data.values()):
                raise TypeError(f"Values within {key} are not integers")

            value = data
            if value["min"] > value["max"]:
                raise ValueError(f"Bad range at {key}")

            output[key] = list(range(value["min"], value["max"] + 1, value["step"]))

        elif isinstance(data, str) or not isinstance(data, abc.Sequence):
            # variable is anything that is not a sequence
            output[key] = [data]
        else:
            # variable is already a sequence
            output[key] = list(data)

    return output
