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
import warnings

from pydantic import BaseModel, ConfigDict, Field, dataclasses, field_validator, model_validator

from corl.environment.default_env_rllib_callbacks import DefaultCallbacks, EnvironmentDefaultCallbacks
from corl.evaluation.runners.section_factories.teams import Teams
from corl.experiments.base_experiment import ExperimentParse
from corl.experiments.rllib_utils.policy_mapping_functions import PolicyIsAgent
from corl.parsers.yaml_loader import load_file


class Config:
    """Pydantic options"""

    arbitrary_types_allowed = True


@dataclasses.dataclass(config=Config)
class Task:
    """Rllib task to be performed"""

    config_yaml_file: str | pathlib.Path | None
    environment: str = "default"
    experiment_parse: ExperimentParse = Field(default=None, validate_default=True)  # type: ignore

    @field_validator("config_yaml_file")
    @classmethod
    def validate_config_yaml_file(cls, v):
        """Ensure config_yaml_file is a pathlib.Path"""
        return pathlib.Path(v)

    @field_validator("experiment_parse", mode="before")
    def validate_experiment_parse(cls, v, values):
        """load experiment_parse"""
        if v is None:
            assert "config_yaml_file" in values.data
            assert "environment" in values.data
            experiment_config = load_file(config_filename=str(values.data["config_yaml_file"]))
            env_config = experiment_config["config"]["env_config"]
            if isinstance(env_config, dict):
                if values.data["environment"] in env_config:
                    env_config = env_config[values.data["environment"]]
                if isinstance(env_config, list):
                    warnings.warn("Applying patches without the !merge yaml directive is deprecated", DeprecationWarning)
                elif "simulator" not in env_config or "episode_parameter_provider" not in env_config:
                    raise ValueError(f"Invalid experiment config for environment {values.data['environment']}")
                experiment_config["config"]["env_config"] = env_config

            return ExperimentParse(**experiment_config)

        assert isinstance(v, ExperimentParse), "experiment_parse was provided but was not an instance of ExperimentParse"
        return v


class Experiment(BaseModel):
    """configs for running an experiment of the provided task/teams"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: Task
    teams: Teams

    rllib_config: dict
    callbacks: type[DefaultCallbacks] = EnvironmentDefaultCallbacks
    tune_config: dict
    ray_config: dict
    env_config: dict

    @model_validator(mode="before")
    @classmethod
    def validate_experiment(cls, values):
        """Initializes experiment configs using the provided task/teams"""

        experiment_parse = values["task"].experiment_parse
        exp = experiment_parse.experiment_class(**experiment_parse.config)

        # HACK: Ensure that the Simulator generates output AERs for every worker
        exp.config.env_config["simulator"]["config"]["limit_extra_data"] = False

        tmp = exp.create_agents(
            platform_configs=values["teams"].platform_config,
            agent_configs=values["teams"].agent_config,
        )
        exp.config.env_config["agents"], exp.config.env_config["agent_platforms"] = tmp

        tmp = exp.create_environment()
        if hasattr(tmp.simulator, "shutdown"):
            tmp.simulator.shutdown()

        policies, train_policies, _ = exp.create_policies(tmp)

        exp.config.policy_mapping.functor = PolicyIsAgent
        exp.config.policy_mapping.config = {}

        exp.config.rllib_configs["local"]["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": exp.config.policy_mapping.functor(**exp.config.policy_mapping.config),
            "policies_to_train": train_policies,
        }

        values["rllib_config"] = exp.config.rllib_configs["local"]
        values["tune_config"] = exp.config.tune_config
        values["ray_config"] = exp.config.ray_config
        values["env_config"] = exp.config.env_config

        return values
