"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import collections
import copy
import json
import os
import typing
from typing import IO, Any

import ray  # pylint: disable=unused-import # noqa: F401
import yaml
from ray import tune  # pylint: disable=unused-import # noqa: F401
from yaml.constructor import ConstructorError
from yaml.nodes import SequenceNode


class Loader(yaml.SafeLoader):  # pylint: disable=too-few-public-methods,W0223
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

        self._include_mapping: dict = {}

    def construct_python_tuple(self, node):
        """Adds in the capability to process tuples in yaml files
        """
        return tuple(self.construct_sequence(node))

    def construct_sequence(self, node, deep=False):
        """Construct a sequence from a YAML sequence node

        This method extends yaml.constructor.BaseConstructor.construct_sequence by adding support for children with the tag
        `!include-extend`.  Any object with this tag should be constructable to produce a sequence of objects.  Even though
        `!include-extend` is a tag on the child object, the sequence produced by this child is not added as a single element to the sequence
        being produced by this method.  Rather, the output sequence is extended with this list.  Any children with other tags are appended
        into the list in the same manner as yaml.constructor.BaseConstructor.construct_sequence.

        Examples
        --------
        Loader.add_constructor("!include-extend", construct_include)
        with open("primary.yml", "r") as fp:
            config = yaml.load(fp, Loader)

        <file primary.yml>
        root:
          tree1:
            - apple
            - banana
            - cherry
          tree2:
            - type: int
              value: 3
            - type: float
              value: 3.14
            - type: str
              value: pi
          tree3:
            - date
            - elderberry
            - !include-extend secondary.yml
            - mango

        <file secondary.yml>
        - fig
        - grape
        - honeydew
        - jackfruit
        - kiwi
        - lemon

        The output of the code above is:
        config = {
            'root': {
                'tree1': ['apple', 'banana', 'cherry'],
                'tree2': [
                    {'type': 'int', 'value': 3},
                    {'type': 'float', 'value': 3.14},
                    {'type': 'str', 'value': 'pi'}
                ],
                'tree3': ['date', 'elderberry', 'fig', 'grape', 'honeydew', 'jackfruit', 'kiwi', 'lemon', 'mango']
            }
        }
        """
        if not isinstance(node, SequenceNode):
            return super().construct_sequence(node, deep=deep)

        output = []
        for child in node.value:
            this_output = self.construct_object(child, deep=deep)
            if child.tag == '!include-extend':
                if not isinstance(this_output, collections.abc.Sequence):
                    raise ConstructorError(
                        None,
                        None,
                        f"expected a sequence returned by 'include-extend', but found {type(this_output).__name__}",
                        child.start_mark
                    )
                output.extend(this_output)
            else:
                output.append(this_output)

        return output

    def flatten_mapping(self, node):
        merge = []
        index = 0
        while index < len(node.value):
            key_node, value_node = node.value[index]
            if key_node.tag == 'tag:yaml.org,2002:merge':
                del node.value[index]
                if isinstance(value_node, yaml.MappingNode):
                    self.flatten_mapping(value_node)
                    merge.extend(value_node.value)
                elif isinstance(value_node, yaml.SequenceNode):
                    submerge = []
                    for subnode in value_node.value:
                        if not isinstance(subnode, yaml.MappingNode):
                            raise yaml.ConstructorError(
                                "while constructing a mapping",
                                node.start_mark,
                                f"expected a mapping for merging, but found {subnode.id}",
                                subnode.start_mark
                            )
                        self.flatten_mapping(subnode)
                        submerge.append(subnode.value)
                    submerge.reverse()
                    for value in submerge:
                        merge.extend(value)
                else:
                    # TODO FIGURE OUT HOW TO DUMP AND ACCESS THE BASE NODE!!!!
                    # if value_node.tag == '!include-direct':
                    #     filename = os.path.realpath(os.path.join(self._root, self.construct_scalar(value_node)))
                    #     d = yaml.dump(self._include_mapping[filename])
                    #     # for k, v in .items():
                    #     #     sk = yaml.ScalarNode(tag='tag:yaml.org,2002:str', value=str(k))
                    #     #     if isinstance(v, int):
                    #     #         sv = yaml.ScalarNode(tag='tag:yaml.org,2002:int', value=str(v))
                    #     #     else:
                    #     #         sv = yaml.ScalarNode(tag='tag:yaml.org,2002:seq', value=str(v))
                    #     #     merge.extend([(sk, sv)])
                    # else:
                    raise yaml.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"expected a mapping or list of mappings for merging, but found {value_node.id}",
                        value_node.start_mark
                    )
            elif key_node.tag == 'tag:yaml.org,2002:value':
                key_node.tag = 'tag:yaml.org,2002:str'
                index += 1
            else:
                index += 1
        if merge:
            node.value = merge + node.value


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""
    filename = os.path.realpath(os.path.join(loader._root, loader.construct_scalar(node)))  # type: ignore # pylint: disable=protected-access # noqa: E501
    extension = os.path.splitext(filename)[1].lstrip(".")

    with open(filename, "r", encoding="utf-8") as fp:
        if extension in ("yaml", "yml"):  # pylint: disable=no-else-return
            return yaml.load(fp, Loader)
        elif extension in ("json", ):
            return json.load(fp)
        else:
            return "".join(fp.readlines())


def construct_include_direct(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""
    filename = os.path.realpath(os.path.join(loader._root, loader.construct_scalar(node)))  # type: ignore # pylint: disable=protected-access # noqa: E501
    extension = os.path.splitext(filename)[1].lstrip(".")

    with open(filename, "r", encoding="utf-8") as fp:
        if extension in ("yaml", "yml"):  # pylint: disable=no-else-return
            temp = yaml.load(fp, Loader)
            loader._include_mapping[filename] = temp  # pylint: disable=protected-access
            return temp
        elif extension in ("json", ):
            return json.load(fp)
        else:
            return "".join(fp.readlines())


def construct_tune_function(loader: Loader, node: yaml.Node) -> Any:  # pylint: disable=unused-argument
    """Include expression referenced at node."""
    if isinstance(node.value, str) and "tune" in node.value:
        return eval(node.value)  # pylint: disable=eval-used
    return node.value


def construct_include_arr(loader: Loader, node: yaml.Node) -> Any:
    """Identical to above, but accepts an array and appends results as an array."""

    sequence = loader.construct_sequence(node)
    data: typing.List = []
    for item in sequence:
        filename = os.path.abspath(os.path.join(loader._root, item))  # pylint: disable=protected-access
        extension = os.path.splitext(filename)[1].lstrip(".")

        with open(filename, "r", encoding="utf-8") as f:
            if extension in ("yaml", "yml"):  # pylint: disable=no-else-return
                data = data + (yaml.load(f, Loader))
            elif extension in ("json", ):
                data = data + (json.load(f))
            else:
                data = data + ("".join(f.readlines()))  # type: ignore

    return data


Loader.add_constructor("!include", construct_include)
Loader.add_constructor("!include-direct", construct_include_direct)
Loader.add_constructor("!include-extend", construct_include)
Loader.add_constructor("!function", construct_tune_function)
Loader.add_constructor("tag:yaml.org,2002:python/tuple", Loader.construct_python_tuple)
Loader.add_constructor("!include_arr", construct_include_arr)


def load_file(config_filename: str):
    """
    Utility function to load in a specified yaml file
    """
    with open(config_filename, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader)
    return config


def separate_config(config: typing.Dict):
    """
    Utility function to separate the env specific configs from the tune configs
    """
    # we can call ray.init without any arguments so default is no arguments
    ray_config = apply_patches(config.get("ray_config", {}))

    # we must have a tune config or else we cannot call tune.run
    if "tune_config" not in config:
        raise ValueError(f"Could not find a tune_config in {config}")
    tune_config = apply_patches(config["tune_config"])

    # must also get an env_config or else we don't know which environment to run```
    if "env_config" in config:
        env_config = apply_patches(config["env_config"])
    else:
        raise ValueError(f"Could not find a env_config in {config} or rllib_config")

    # must get a rllib config in some way or else we aren't going to run anything useful
    rllib_configs = {}
    if "rllib_configs" in config:
        for key, value in config["rllib_configs"].items():
            rllib_configs[key] = apply_patches(value)
    else:
        raise ValueError(f"Could not find a rllib_config in {config} or 'config' in tune_config")

    for key, value in rllib_configs.items():
        value["env_config"] = copy.deepcopy(env_config)
        value["env_config"].setdefault("environment", {})["horizon"] = value.get("horizon", 1000)

    # a trainable config is not necessary
    trainable_config = apply_patches(config.get("trainable_config", None))

    return ray_config, rllib_configs, tune_config, trainable_config


def apply_patches(config):
    """updates the base setup with patches

    Arguments:
        config [dict, list] -- The base and patch if list, else dict

    Returns:
        The combined dict
    """

    def merge(source, destination):
        """
        run me with nosetests --with-doctest file.py

        >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
        >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
        >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
        True
        """
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                merge(value, node)
            else:
                destination[key] = value

        return destination

    if isinstance(config, list):
        config_new = copy.deepcopy(config[0])
        for item in config[1:]:
            if item is not None:
                config_new = merge(item, config_new)
        return config_new

    return config
