"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import abc
import typing
from collections.abc import MutableMapping

from pydantic import BaseModel, FilePath, ImportString, TypeAdapter, field_validator

from corl.agents.base_agent import AgentParseBase, AgentParseInfo
from corl.libraries.file_path import CorlDirectoryPath
from corl.parsers.agent_and_platform import CorlAgentConfigArgs, CorlPlatformConfigArgs
from corl.parsers.yaml_loader import load_file
from corl.policies.base_policy import BasePolicyValidator


class ExperimentFileParse(BaseModel):
    """[summary]
    experiment_class: The experiment class to run
    config: the configuration to pass to that experiment
    """

    config: FilePath
    compute_platform: str = "local"
    platform_config: list[CorlPlatformConfigArgs]
    agent_config: list[CorlAgentConfigArgs]
    environment: str = "default"
    name: str | None = None
    output: CorlDirectoryPath | None = None
    debug: bool = False


class BaseExperimentValidator(BaseModel):
    """
    Base Validator to subclass for Experiments subclassing BaseExperiment
    """


class BaseExperiment(abc.ABC):
    """
    Experiment provides an anstract class to run specific types of experiments
    this allows users to do specific setup steps or to run some sort of custom training
    loop
    """

    def __init__(self, **kwargs) -> None:
        self.config: BaseExperimentValidator = self.get_validator()(**kwargs)

    @staticmethod
    def get_validator() -> type[BaseExperimentValidator]:
        """
        Get the validator for this experiment class,
        the kwargs sent to the experiment class will
        be validated using this object and add a self.config
        attr to the experiment class
        """
        return BaseExperimentValidator

    @property
    def get_policy_validator(self) -> type[BasePolicyValidator]:
        """
        Get the policy validator for this experiment class,
        the kwargs sent to the experiment class will
        be validated using this object and add a self.config
        attr to the policy config
        """
        return BasePolicyValidator

    @abc.abstractmethod
    def run_experiment(self, args: ExperimentFileParse):
        """
        Runs the experiment associated with this experiment class

        Arguments:
            args {ExperimentFileParse} -- The args provided by the argparse
                                            in corl.train_rl
        """

    def create_agents(self, platform_configs: list[CorlPlatformConfigArgs], agent_configs: list[CorlAgentConfigArgs]) -> tuple[dict, dict]:
        """Create the requested agents and add them to the environment configuration.

        Parameters
        ----------
        platform_configs : CorlPlatformsConfigArg
            A sequence of platforms
        agent_configs : agent_configs
            A sequence of agents
        """
        platforms = {}
        for platform_config in platform_configs:
            assert platform_config.name not in platforms, "duplicate platforms not allowed"
            platform_configuration = platform_config.config

            platforms[platform_config.name] = platform_configuration

        agents = {}
        for agent_config in agent_configs:
            if not all(platform in platforms for platform in agent_config.platforms):
                raise RuntimeError(
                    f"invalid platform request by agent {agent_config.name}, "
                    f"agent requested platforms {agent_config.platforms}, but available platforms are: "
                    f"{platforms}"
                )

            config = agent_config.config
            policy_config = agent_config.policy

            parsed_agent = AgentParseBase(**config)
            _ = self.get_policy_validator(**policy_config)

            agents[agent_config.name] = AgentParseInfo(
                class_config=parsed_agent, platform_names=agent_config.platforms, policy_config=policy_config
            )

        return agents, platforms

    @staticmethod
    def create_other_platforms(other_platforms_config: typing.Sequence[tuple[str, str]]) -> dict:
        """Create the requested other platforms and add them to the environment configuration.

        Parameters
        ----------
        other_platforms_config : typing.Sequence[typing.Tuple[str, str]]
            A sequence of platforms.  Each platform consists of a name and platform filename.
        """
        other_platforms = {}

        if other_platforms_config:
            for platform_name, platform_file in other_platforms_config:
                platform_config = load_file(platform_file)
                other_platforms[platform_name] = platform_config
        return other_platforms

    @staticmethod
    def _update_config_object(target: dict, path: typing.Sequence[str], arg, func):
        """
        Updates/Modifies a config mapping object.

        target:  Config mapping object
        path:    List of key words identifying where in the target to modify a value
        arg:     Value that will modify or replace a value of target (i.e. 'output')
        func:    Functor that performs action on data in arg (i.e. lambda x : os.path.join(x,'a'))
        """

        def _build_config_dict_entry(local_path, value):
            """
            Given a path, recursively generate a new dict or dict-of-dicts entry for target.
            """
            if len(local_path) == 1:
                return {local_path[0]: value}

            return {local_path.pop(0): _build_config_dict_entry(local_path, value)}

        if isinstance(target, list):
            target.append(_build_config_dict_entry(path, func(arg)))
        elif isinstance(target, MutableMapping):
            embedded_dict = target
            for key in path:
                if key == path[-1]:
                    embedded_dict[key] = func(arg)
                else:
                    embedded_dict = embedded_dict[key]
        else:
            raise TypeError(f"Unsupported type: {type(target).__name__}")

    @staticmethod
    def process_cli_args(config: dict, cli_args: ExperimentFileParse):
        """Modifies/overrides config fields with command line arguments."""


class ExperimentParse(BaseModel):
    """[summary]
    experiment_class: The experiment class to run
    config: the configuration to pass to that experiment
    """

    experiment_class: type[BaseExperiment]
    config: dict[str, typing.Any]

    @field_validator("experiment_class", mode="before")
    @classmethod
    def construct_experiment_class(cls, v):
        """Convert string to callable"""
        if not isinstance(v, type):
            return TypeAdapter(ImportString).validate_python(v)
        return v
