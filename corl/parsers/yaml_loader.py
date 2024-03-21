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
from pathlib import Path
from typing import IO, Any
from urllib.parse import unquote, urlparse

import importlib_metadata
import yaml
from ray import tune
from yaml.constructor import ConstructorError
from yaml.nodes import SequenceNode


def get_pkgs_paths() -> set[Path]:
    """returns set of full paths to all pkgs"""
    pkgs_paths = set()
    for distribution in importlib_metadata.distributions():
        if distribution.origin:
            pkgs_paths.add(Path(unquote(urlparse(distribution.origin.url).path)))
        else:
            pkgs_paths.add(Path(str(distribution.locate_file(""))))
    return pkgs_paths


def get_distribution_path(distribution: importlib_metadata.Distribution) -> Path:
    """returns set of full path to top level of a package"""
    dist_name = importlib_metadata.Prepared.normalize(distribution.name)
    if distribution.origin:
        return Path(unquote(urlparse(distribution.origin.url).path)) / Path(dist_name)
    return (Path(str(distribution.locate_file("")))) / Path(dist_name)


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self.root_path = Path(os.path.split(stream.name)[0])
        except AttributeError:
            self.root_path = Path(os.path.curdir)

        self.cwd_path = Path.cwd()

        self._pkg_paths = get_pkgs_paths()
        self._distribution = None

        for site_path in self._pkg_paths:
            if site_path in self.root_path.parents:
                str(Path(*self.root_path.relative_to(site_path).parts[:1]))
                if site_path != self.cwd_path:
                    self._distribution = importlib_metadata.distribution(str(Path(*self.root_path.relative_to(site_path).parts[:1])))

        super().__init__(stream)

        self._include_mapping: dict = {}
        self.deep_construct = True

    def build_file_path(self, node, absolute: bool = False):
        """
        handles loading the strings behind the various corl !include
        extensions to handle complex loading

        corl performs a prioritized search in the following order to
        locate a file and build the associated path. The first search
        that returns a valid file will be return the path. Subsequent
        searches will not be performed.

        1. Relative paths
        if an include filepath begins with a '.' the loader will use the file location as a basis
        no searches are performed beyond relative locations of current file.
        file path: config/tasks/docking1d/agents/main_agent.yml
        example1: !include ./glues/glue_set1.yml
        would be resolved to config/tasks/docking1d/agents/glues/glue_set1.yml

        2. CWD paths
        corl will look for files pathed from the current working directory
        example: !include config/tasks/docking1d/agents/glue_set1.yml
        would be resolved to load <path to working directory>/config/tasks/docking1d/agents/glue_set1.yml

        3. Module paths
        corl sets a module path for each yaml conf. The module path is set to the cwd
        path, unless the current yml conf resides in a site_pkgs (i.e. is installed).
        In that case the module path will be set the the site package directory of the module.
        example: /home/user/src/corl/config/tasks/docking1d/agents/glue_set1.yml
        module_path: cwd
        example: opt/conda/lib/python3.10/site-packages/<pkg>/config/tasks/docking1d/agents/glue_set1.yml
        module_path: opt/conda/lib/python3.10/site-packages/<pkg>

        when parsing a module input config include path can now be defined relative the module
        for example in the config opt/conda/lib/python3.10/site-packages/<pkg>/config/sample_config1.yml
        example: !include config/sample_config2.yml
        would be resolved to load opt/conda/lib/python3.10/site-packages/<pkg>/config/sample_config2.yml

        4. Site-package paths
        corl will search site package for includes not found in other paths
        example: !include <pkg>/config/sampple_config1.yml
        would be resolved to load opt/conda/lib/python3.10/<pkg>/config/sampple_config1.yml
        """
        loader_string = self.construct_scalar(node)
        assert isinstance(loader_string, str)

        filename, distribution = self._find_file(loader_string)

        if not filename:
            raise RuntimeError(f"{loader_string} not found when yaml parsing")

        return filename.absolute() if absolute else filename.resolve(), distribution

    def _find_file(self, loader_string) -> tuple[Path | None, importlib_metadata.Distribution | None]:
        """ """
        # Check if file is relative to current file
        if loader_string.startswith("."):
            tmp_path = Path(self.root_path, loader_string)
            if tmp_path.is_file():
                return tmp_path, self._distribution
            return None, None

        # Check if file is in current repo
        cwd_path = Path(self.cwd_path, loader_string)
        if cwd_path.is_file():
            return cwd_path, None

        # Check if file is in current module
        if self._distribution:
            module_path = Path(get_distribution_path(self._distribution), loader_string)
            if module_path.is_file():
                return module_path, self._distribution

        # Check if file is in a different site pkg module
        for pkg_path in self._pkg_paths:
            site_path = Path(pkg_path, loader_string)
            if site_path.is_file():
                return site_path, importlib_metadata.distribution(str(Path(*site_path.relative_to(pkg_path).parts[:1])))

        return None, None

    def construct_str(self, node):
        # Implement custom string handling here
        if Path(node.value).suffix:
            new_file_path, _ = self._find_file(node.value)
            if new_file_path:
                return str(new_file_path)
        return node.value

    def construct_document(self, node):
        data = super().construct_document(node)
        self.deep_construct = True
        return data

    def construct_python_tuple(self, node):
        """Adds in the capability to process tuples in yaml files"""
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
            if child.tag == "!include-extend":
                if not isinstance(this_output, collections.abc.Sequence):
                    raise ConstructorError(
                        None,
                        None,
                        f"expected a sequence returned by 'include-extend', but found {type(this_output).__name__}",
                        child.start_mark,
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
            if key_node.tag == "tag:yaml.org,2002:merge":
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
                                subnode.start_mark,
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
                        value_node.start_mark,
                    )
            elif key_node.tag == "tag:yaml.org,2002:value":
                key_node.tag = "tag:yaml.org,2002:str"
                index += 1
            else:
                index += 1
        if merge:
            node.value = merge + node.value


def include_file(filename: Path) -> Any:
    extension = filename.suffix

    with open(filename, encoding="utf-8") as fp:
        if extension in (".yaml", ".yml"):
            return yaml.load(fp, Loader)  # noqa: S506
        elif extension in (".json",):  # noqa: RET505
            return json.load(fp)
        else:
            return "".join(fp.readlines())


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""
    filename, _ = loader.build_file_path(node)
    return include_file(filename)


def construct_path(loader: Loader, node: yaml.Node) -> Any:
    """Construct file path associated with node"""
    filename, _ = loader.build_file_path(node)
    return filename


def construct_include_direct(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""
    filename, file_module = loader.build_file_path(node)
    extension = filename.suffix

    with open(filename, encoding="utf-8") as fp:
        if extension in (".yaml", ".yml"):
            temp = yaml.load(fp, Loader)  # noqa: S506
            loader._include_mapping[filename] = temp  # noqa: SLF001
            return temp
        elif extension in (".json",):  # noqa: RET505
            return json.load(fp)
        else:
            return "".join(fp.readlines())


def construct_tune_function(loader: Loader, node: yaml.Node) -> Any:
    """Include expression referenced at node."""
    if isinstance(node.value, str) and "tune" in node.value:
        return eval(node.value)  # noqa: S307
    return node.value


def construct_include_arr(loader: Loader, node: yaml.Node) -> Any:
    """Identical to above, but accepts an array and appends results as an array."""

    sequence = loader.construct_sequence(node)
    data: list = []
    for item in sequence:
        filename, _ = loader.build_file_path(item, absolute=True)
        data = data + include_file(filename)
    return data


def construct_grid_search(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""
    sequence = loader.construct_sequence(node)
    return tune.grid_search(sequence)


def construct_merge_dict(loader: Loader, node: yaml.Node):
    sequence = loader.construct_sequence(node)
    return apply_patches(sequence)


Loader.add_constructor("!include", construct_include)
Loader.add_constructor("!include-direct", construct_include_direct)
Loader.add_constructor("!include-extend", construct_include)
Loader.add_constructor("!function", construct_tune_function)
Loader.add_constructor("tag:yaml.org,2002:python/tuple", Loader.construct_python_tuple)
Loader.add_constructor("tag:yaml.org,2002:str", Loader.construct_str)
Loader.add_constructor("!include_arr", construct_include_arr)
Loader.add_constructor("!merge", construct_merge_dict)
Loader.add_constructor("!tune_grid_search", construct_grid_search)
Loader.add_constructor("!corl_path", construct_path)


def load_file(config_filename: str | Path):
    """
    Utility function to load in a specified yaml file
    """
    with open(config_filename, encoding="utf-8") as fp:
        return yaml.load(fp, Loader)  # noqa: S506


def separate_config(config: dict):
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

    for value in rllib_configs.values():
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
