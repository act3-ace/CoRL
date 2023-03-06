"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Defines a task
"""
import pathlib
import typing

from pydantic import BaseModel, dataclasses, root_validator, validator

from corl.environment.default_env_rllib_callbacks import DefaultCallbacks, EnvironmentDefaultCallbacks
from corl.environment.multi_agent_env import ACT3MultiAgentEnv
from corl.evaluation.runners.section_factories.teams import Teams
from corl.experiments.base_experiment import ExperimentParse
from corl.parsers.yaml_loader import load_file


class Config:
    """Pydantic options"""
    arbitrary_types_allowed = True


@dataclasses.dataclass(config=Config)
class Task:
    """Rllib task to be performed
    """

    config_yaml_file: typing.Optional[typing.Union[str, pathlib.Path]]
    experiment_parse: typing.Optional[ExperimentParse] = None

    @validator('config_yaml_file')
    def validate_config_yaml_file(cls, v):
        """Ensure config_yaml_file is a pathlib.Path"""
        return pathlib.Path(v)

    @validator('experiment_parse')
    def validate_experiment_parse(cls, v, values):
        """load experiment_parse"""
        if not v:
            experiment_config = load_file(config_filename=str(values['config_yaml_file']))
            experiment_parse = ExperimentParse(**experiment_config)
            return experiment_parse

        assert not values['config_yaml_file'], 'Invalid configuration - only 1 of [config_yaml_file, experiment_parse] may be defined'
        return v


class Experiment(BaseModel):
    """configs for running an experiment of the provided task/teams
    """

    class Config:
        """Pydantic options"""
        arbitrary_types_allowed = True

    task: Task
    teams: Teams

    rllib_config: dict
    callbacks: typing.Type[DefaultCallbacks] = EnvironmentDefaultCallbacks
    tune_config: dict
    ray_config: dict
    env_config: dict
    trainable_config: typing.Optional[dict]

    @root_validator(pre=True)
    def validate_experiment(cls, values):
        """Initializes experiment configs using the provided task/teams"""

        experiment_parse = values['task'].experiment_parse
        exp = experiment_parse.experiment_class(**experiment_parse.config)
        tmp = exp.create_agents(
            platform_configs=values['teams'].platform_config,
            agent_configs=values['teams'].agent_config,
        )
        exp.config.env_config["agents"], exp.config.env_config["agent_platforms"] = tmp

        tmp = ACT3MultiAgentEnv(exp.config.env_config)
        tmp_as = tmp.action_space
        tmp_os = tmp.observation_space
        tmp_ac = exp.config.env_config["agents"]

        policies = {
            policy_name: (
                tmp_ac[policy_name].policy_config["policy_class"],
                policy_obs,
                tmp_as[policy_name],  # pylint: disable=unsubscriptable-object
                tmp_ac[policy_name].policy_config["config"]
            )
            for policy_name,
            policy_obs in tmp_os.spaces.items()
            if tmp_ac[policy_name]
        }

        train_policies = [policy_name for policy_name in policies.keys() if tmp_ac[policy_name].policy_config["train"]]
        exp.config.rllib_configs['local']["multiagent"] = {
            "policies": policies, "policy_mapping_fn": lambda agent_id: agent_id, "policies_to_train": train_policies
        }

        values['rllib_config'] = exp.config.rllib_configs['local']
        values['tune_config'] = exp.config.tune_config
        values['ray_config'] = exp.config.ray_config
        values['env_config'] = exp.config.env_config
        values['trainable_config'] = exp.config.trainable_config

        return values
