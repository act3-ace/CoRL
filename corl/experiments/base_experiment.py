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
import argparse
import typing

from pydantic import BaseModel, PyObject, validator

from corl.agents.base_agent import AgentParseBase, AgentParseInfo
from corl.parsers.yaml_loader import load_file
from corl.policies.base_policy import BasePolicyValidator


class BaseAutoDetect:
    """Base class interface for setting rllib config if in auto mode
    """

    def autodetect_system(self) -> str:
        """gets the default system based on user defined function

        Returns
        -------
        str
            the base system to use
        """
        return "local"


class BaseExperimentValidator(BaseModel):
    """
    Base Validator to subclass for Experiments subclassing BaseExperiment
    """
    ...


class BaseExperiment(abc.ABC):
    """
    Experiment provides an anstract class to run specific types of experiments
    this allows users to do specific setup steps or to run some sort of custom training
    loop
    """

    def __init__(self, **kwargs) -> None:
        self.config: BaseExperimentValidator = self.get_validator(**kwargs)

    @property
    def get_validator(self) -> typing.Type[BaseExperimentValidator]:
        """
        Get the validator for this experiment class,
        the kwargs sent to the experiment class will
        be validated using this object and add a self.config
        attr to the experiment class
        """
        return BaseExperimentValidator

    @property
    def get_policy_validator(self) -> typing.Type[BasePolicyValidator]:
        """
        Get the policy validator for this experiment class,
        the kwargs sent to the experiment class will
        be validated using this object and add a self.config
        attr to the policy config
        """
        return BasePolicyValidator

    @abc.abstractmethod
    def run_experiment(self, args: argparse.Namespace):
        """
        Runs the experiment associated with this experiment class

        Arguments:
            args {argparse.Namespace} -- The args provided by the argparse
                                            in corl.train_rl
        """
        ...

    def create_agents(
        self, platform_configs: typing.Sequence[typing.Tuple[str, str]], agent_configs: typing.Sequence[typing.Tuple[str, str, str, str]]
    ) -> typing.Tuple[dict, dict]:
        """Create the requested agents and add them to the environment configuration.

        Parameters
        ----------
        agent_configs : typing.Sequence[typing.Tuple[str, str, str, str]]
            A sequence of agents.  Each agent consists of a name, configuration filename, platform filename
            and policy configuration filename.
        """
        platforms = {}
        for platform_name, platform_file in platform_configs:
            assert platform_name not in platforms, 'duplicate platforms not allowed'
            platform_config = load_file(platform_file)
            platforms[platform_name] = platform_config

        agents = {}
        for policy_name, platform_name, agent_file, policy_file in agent_configs:
            assert platform_name in platforms, f"invalid platform '{platform_name}' not in {platforms}"

            config = load_file(agent_file)
            parsed_agent = AgentParseBase(**config)
            policy_config = load_file(policy_file)
            parsed_policy = self.get_policy_validator(**policy_config)
            agents[policy_name] = AgentParseInfo(class_config=parsed_agent, platform_name=platform_name, policy_config=parsed_policy)

        return agents, platforms

    @staticmethod
    def create_other_platforms(other_platforms_config: typing.Sequence[typing.Tuple[str, str]]) -> dict:
        """Create the requested other platforms and add them to the environment configuration.

        Parameters
        ----------
        other_platforms_config : typing.Sequence[typing.Tuple[str, str]]
            A sequence of platforms.  Each platform consists of a name and platform filename.
        """
        other_platforms = dict()

        if other_platforms_config:
            for platform_name, platform_file in other_platforms_config:
                platform_config = load_file(platform_file)
                other_platforms[platform_name] = platform_config
        return other_platforms


class ExperimentParse(BaseModel):
    """[summary]
    experiment_class: The experiment class to run
    config: the configuration to pass to that experiment
    """
    experiment_class: PyObject
    auto_system_detect_class: typing.Optional[PyObject] = None
    config: typing.Dict[str, typing.Any]

    @validator('experiment_class')
    def check_experiment_class(cls, v):
        """
        Validates the experiment class actually subclasses BaseExperiment Class
        """
        if not issubclass(v, BaseExperiment):
            raise ValueError(f"Experiment functors must subclass BaseExperiment, but experiment {v}")
        return v

    @validator('auto_system_detect_class')
    def check_auto_system_detect_class(cls, v):
        """
        Validates the auto system detect class actually subclasses BaseAutoDetect Class
        """
        if v is not None:
            if not issubclass(v, BaseAutoDetect):
                raise ValueError(f"Experiment functors must subclass BaseAutoDetect, but experiment {v}")
        return v
