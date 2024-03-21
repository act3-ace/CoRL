"""
---------------------------------------------------------------------------.

Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import json
import re
import typing
from collections import OrderedDict

import flatten_dict
import pandas as pd
from gymnasium.spaces.dict import Dict
from numpy import ndarray
from pydantic import BaseModel, ConfigDict, ImportString, field_validator
from ray.rllib.utils.spaces.repeated import Repeated

from corl.environment.multi_agent_env import ACT3MultiAgentEnv
from corl.experiments.base_experiment import BaseExperiment, BaseExperimentValidator, ExperimentFileParse
from corl.libraries.env_space_util import SingleLayerDict
from corl.libraries.units import Quantity
from corl.parsers.yaml_loader import apply_patches
from corl.policies.base_policy import BasePolicyValidator


class QuantityEncoder(json.JSONEncoder):
    """Basic encoder of quantity variables

    Args:
        json (_type_): _description_
    """

    def default(self, obj):
        if isinstance(obj, Quantity):
            if isinstance(obj.m, ndarray):
                return obj.m.tolist(), str(obj.unit)
            return obj.m, str(obj.unit)
        if isinstance(obj, BaseModel):
            return dict(obj)
        return json.JSONEncoder.default(self, obj)


class ExportSetupExperimentValidator(BaseExperimentValidator):
    """
    ray_config: dictionary to be fed into ray init, validated by ray init call
    env_config: environment configuration, validated by environment class
    rllib_configs: a dictionary
    Arguments:
        BaseModel {[type]} -- [description].

    Raises
    ------
        RuntimeError: [description]

    Returns
    -------
        [type] -- [description]
    """

    ray_config: dict[str, typing.Any]
    env_config: dict[str, typing.Any]
    rllib_configs: dict[str, dict[str, typing.Any]]
    tune_config: dict[str, typing.Any]
    trainable_config: dict[str, typing.Any] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("rllib_configs", mode="before")
    @classmethod
    def apply_patches_rllib_configs(cls, v):
        """
        The dictionary of rllib configs may come in as a dictionary of
        lists of dictionaries, this function is responsible for collapsing
        the list down to a typing.Dict[str, typing.Dict[str, typing.Any]]
        instead of
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]]].

        Raises
        ------
        RuntimeError: [description]

        Returns
        -------
        [type] -- [description]
        """
        if not isinstance(v, dict):
            raise TypeError("rllib_configs are expected to be a dict of keys to different compute configs")
        rllib_configs = {}
        for key, value in v.items():
            if isinstance(value, list):
                rllib_configs[key] = apply_patches(value)
            elif isinstance(value, dict):
                rllib_configs[key] = value
        return rllib_configs

    @field_validator("ray_config", "tune_config", "trainable_config", "env_config", mode="before")
    @classmethod
    def apply_patches_configs(cls, v):
        """
        reduces a field from
        typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]]]
        to
        typing.Dict[str, typing.Any].

        by patching the first dictionary in the list with each patch afterwards

        Returns
        -------
        [type] -- [description]
        """
        if isinstance(v, list):
            v = apply_patches(v)
        return v


class RllibPolicyValidator(BasePolicyValidator):
    """
    policy_class: callable policy class None will use default from trainer
    train: should this policy be trained
    Arguments:
        BaseModel {[type]} -- [description].

    Raises
    ------
        RuntimeError: [description]

    Returns
    -------
        [type] -- [description]
    """

    config: dict[str, typing.Any] = {}
    policy_class: ImportString | None = None
    train: bool = True


class ExportSetupExperiment(BaseExperiment):
    """
    The Rllib Experiment is an experiment for running
    multi agent configurable environments with patchable settings.
    """

    def __init__(self, **kwargs) -> None:
        self.config: ExportSetupExperimentValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[ExportSetupExperimentValidator]:
        return ExportSetupExperimentValidator

    @property
    def get_policy_validator(self) -> type[RllibPolicyValidator]:
        """Return validator."""
        return RllibPolicyValidator

    def run_experiment(self, args: ExperimentFileParse) -> None:  # noqa: PLR0915
        self.setup_configs(args)

        tmp: ACT3MultiAgentEnv = ACT3MultiAgentEnv(self.config.env_config)
        tmp.reset()
        tmp.step(tmp.action_space.sample())

        def remove_duplicates(in_str) -> str:
            """Remove duplicates"""
            # split input string separated by space
            in_str_list = in_str.split(" ")

            # now create dictionary using counter method
            # which will have strings as key and their
            # frequencies as value
            UniqW = list(OrderedDict.fromkeys(in_str_list))

            return " ".join(UniqW)

        def filter_string(k) -> str:
            """__summary_"""
            temp_string = k.replace("_", " ")
            temp_string = temp_string.replace(".", " . ")
            temp_string = re.sub(r"(\w)([A-Z])", r"\1 \2", temp_string)
            temp_string = temp_string.title()
            temp_string = " ".join(re.findall("[A-Z][^A-Z]*", temp_string)).title()
            temp_string = re.sub(" +", " ", temp_string)
            temp_string = remove_duplicates(temp_string)
            temp_string = (
                temp_string.replace(" ", "").replace("ObsSensor", "").replace("ObserveSensorRepeated", "").replace("DirectObservation", "")
            )
            return temp_string[:-1] if temp_string[-1] == "." else temp_string

        def get_dataframe(in_space):
            if issubclass(type(in_space), dict | Dict | SingleLayerDict):
                temp_space = flatten_dict.flatten(in_space, reducer="dot")
            else:
                temp_space = {"control": in_space}

            temp_new_space = OrderedDict()
            for k, v in temp_space.items():
                if isinstance(v, Repeated):
                    rep_space = flatten_dict.flatten(v.child_space, reducer="dot")
                    for o_k, o_v in rep_space.items():
                        temp_new_space[f"{k}.{o_k}"] = o_v
                elif isinstance(k, tuple):
                    temp_new_space[".".join(k)] = v
                else:
                    temp_new_space[k] = v

            filtered_temp_space = {
                filter_string(flattened_key): str(flatten_value) for flattened_key, flatten_value in temp_new_space.items()
            }
            return pd.DataFrame(data=filtered_temp_space, index=[0])

        def process_space(in_space, in_space_str, in_space_units, experiment_config_id, experiment_path, writer):
            df_temp_space = get_dataframe(in_space)
            df_temp_space_units = get_dataframe(in_space_units)
            df_temp_space = pd.concat([df_temp_space, df_temp_space_units])
            df_temp_space = df_temp_space.fillna("Not In Space").T
            df_temp_space.columns = ["Space", "Unit"]
            df_temp_space.index.name = "Parameter"
            if experiment_config_id and experiment_path:
                writer.write("\n")
                writer.write(df_temp_space.to_markdown())
                writer.write("\n\n")

        if hasattr(self, "experiment_config_id"):
            experiment_config_id = self.experiment_config_id
            experiment_path = self.experiment_path  # type: ignore
        else:
            experiment_config_id = None
            experiment_path = None

        def write_details_start(writer, string):
            writer.write("\n")
            writer.write(f"/// details | Expand to see the {string}\n")
            writer.write("    type: note\n")
            writer.write("    open: false\n")
            writer.write("\n")

        def write_details_end(writer):
            writer.write("\n\n")
            writer.write("///\n\n")

        # Process the observations and actions for each agen
        #
        file_path = f"{experiment_path}/{experiment_config_id}.md"
        with open(file_path, "w") as writer:
            for agent_key, agent_value in tmp._agent_dict.items():  # noqa: SLF001
                # for debug runs with both agents the same just skip the red policies
                writer.write(f"#### {agent_key[agent_key.find('_') + 1 :].upper()} Agent\n")
                writer.write("\n")

                # reward_list = agent_value.agent_reward_functors
                # if reward_list:
                #     write_details_start(writer, "Rewards")
                #     df_reward = get_dataframe({r.name: print_fields(dict(r.config)) for r in reward_list}).T
                #     writer.write(df_reward.T.to_markdown())
                #     write_details_end(writer)

                write_details_start(writer, "Action Space")
                process_space(
                    agent_value.action_space, "Action Space", agent_value.action_units, experiment_config_id, experiment_path, writer
                )
                write_details_end(writer)

                write_details_start(writer, "Observation Space")
                process_space(
                    agent_value.observation_space,
                    "Observation Space",
                    agent_value.observation_units,
                    experiment_config_id,
                    experiment_path,
                    writer,
                )
                write_details_end(writer)

                write_details_start(writer, "Normalized Action Space")
                process_space(
                    agent_value.normalized_action_space,
                    "Action Space",
                    agent_value.action_units,
                    experiment_config_id,
                    experiment_path,
                    writer,
                )
                write_details_end(writer)

                write_details_start(writer, "Normalized Observation Space")
                process_space(
                    agent_value.normalized_observation_space,
                    "Observation Space",
                    agent_value.observation_units,
                    experiment_config_id,
                    experiment_path,
                    writer,
                )
                write_details_end(writer)

            # Process the evnironment dones
            writer.write("#### Task Termination / Goal Criteria -\n")
            writer.write("\n")

            for index, value in enumerate(tmp._shared_done.process_callbacks):  # noqa: SLF001
                writer.write(f"/// tab | {value.name}\n")  # type: ignore
                if index == 0:
                    writer.write("    new: true\n")
                writer.write("\n")
                writer.write("```yaml\n")
                writer.write(json.dumps(dict(value.config), indent=4, sort_keys=True, cls=QuantityEncoder))  # type: ignore
                writer.write("\n")
                writer.write("```\n")
                writer.write("\n")
                writer.write("///\n\n")

            writer.write("#### Platform Termination / Goal Criteria -\n")
            writer.write("\n")

            for key, dones in tmp._done.items():  # noqa: SLF001
                writer.write(f"##### {key} -\n")
                writer.write("\n")

                for index, value in enumerate(dones):
                    writer.write(f"/// tab | {value.name}\n")  # type: ignore
                    if index == 0:
                        writer.write("    new: true\n")
                    writer.write("\n")
                    writer.write("```yaml\n")
                    writer.write(json.dumps(dict(value.config), indent=4, sort_keys=True, cls=QuantityEncoder))  # type: ignore
                    writer.write("\n")
                    writer.write("```\n")
                    writer.write("\n")
                    writer.write("///\n\n")

    def setup_configs(self, args):
        """
        tbd
        """
        rllib_config = self._select_rllib_config(args.compute_platform)
        if args.compute_platform in ["ray"]:
            self._update_ray_config_for_ray_platform()

        self.config.env_config["agents"], self.config.env_config["agent_platforms"] = self.create_agents(
            args.platform_config,
            args.agent_config,
        )

        self.config.env_config["horizon"] = rllib_config["horizon"]

    def _select_rllib_config(self, platform: str | None) -> dict[str, typing.Any]:
        """Extract the rllib config for the proper computational platform.

        Parameters
        ----------
        platform : typing.Optional[str]
            Specification of the computational platform to use, such as "local", "hpc", etc.  This must be present in the rllib_configs.
            If None, the rllib_configs must only have a single entry.

        Returns
        -------
        typing.Dict[str, typing.Any]
            Rllib configuration for the desired computational platform.

        Raises
        ------
        RuntimeError
            The requested computational platform does not exist or None was used when multiple platforms were defined.
        """
        if platform is not None:
            return self.config.rllib_configs[platform]

        if len(self.config.rllib_configs) == 1:
            return self.config.rllib_configs[next(iter(self.config.rllib_configs))]

        raise RuntimeError(f'Invalid rllib_config for platform "{platform}"')

    def _update_ray_config_for_ray_platform(self) -> None:
        """Update the ray configuration for ray platforms."""
        self.config.ray_config["address"] = "auto"
        self.config.ray_config["log_to_driver"] = False
