"""
---------------------------------------------------------------------------.

Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import os
import pathlib
import time
import typing

import ray
from pydantic import ConfigDict, ImportString, field_validator
from pyinstrument import Profiler
from ray.tune.registry import get_trainable_cls

from corl.environment.default_env_rllib_callbacks import EnvironmentDefaultCallbacks
from corl.environment.multi_agent_env import ACT3MultiAgentEnv, ACT3MultiAgentEnvValidator
from corl.episode_parameter_providers import EpisodeParameterProvider
from corl.episode_parameter_providers.remote import RemoteEpisodeParameterProvider
from corl.experiments.base_experiment import BaseExperiment, BaseExperimentValidator, ExperimentFileParse
from corl.libraries.factory import Factory
from corl.parsers.yaml_loader import apply_patches
from corl.policies.base_policy import BasePolicyValidator


class BenchmarkExperimentValidator(BaseExperimentValidator):
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
    benchmark_episodes: int = 10

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


class BenchmarkExperiment(BaseExperiment):
    """
    The Rllib Experiment is an experiment for running
    multiagent configurable environments with patchable settings.
    """

    def __init__(self, **kwargs) -> None:
        self.config: BenchmarkExperimentValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[BenchmarkExperimentValidator]:
        return BenchmarkExperimentValidator

    @property
    def get_policy_validator(self) -> type[RllibPolicyValidator]:
        """Return validator."""
        return RllibPolicyValidator

    def run_experiment(self, args: ExperimentFileParse) -> float:
        rllib_config = self._select_rllib_config(args.compute_platform)
        if args.compute_platform in ["ray"]:
            self._update_ray_config_for_ray_platform()

        if args.debug:
            rllib_config["num_workers"] = 0
            self.config.ray_config["local_mode"] = True

        self._add_trial_creator()

        ray.init(**self.config.ray_config)

        self.config.env_config["agents"], self.config.env_config["agent_platforms"] = self.create_agents(
            args.platform_config,
            args.agent_config,
        )

        # self.config.env_config["horizon"] = rllib_config["horizon"]

        if not self.config.ray_config["local_mode"]:
            self.config.env_config["episode_parameter_provider"] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                Factory(**self.config.env_config["episode_parameter_provider"]),
                actor_name=ACT3MultiAgentEnv.episode_parameter_provider_name,
            )

            for agent_name, agent_configs in self.config.env_config["agents"].items():
                agent_configs.class_config.config["episode_parameter_provider"] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                    Factory(**agent_configs.class_config.config["episode_parameter_provider"]),
                    agent_name,
                )

        self.config.env_config["epp_registry"] = ACT3MultiAgentEnvValidator(**self.config.env_config).epp_registry

        tmp = ACT3MultiAgentEnv(self.config.env_config)
        act_space = tmp.action_space

        env = tmp

        def save_episode_information(_, __, ___, ____):
            pass

        env._simulator.save_episode_information = save_episode_information  # type: ignore  # noqa: SLF001
        env.config.deep_sanity_check = True
        profiler = Profiler()
        profiler.start()

        # retrieve action
        # if sanity_check_state_dict:
        total_timesteps = 0
        t1 = time.time()

        for _ep in range(self.config.benchmark_episodes):
            print(f"Episode {_ep}")
            time.time()
            env.reset()

            # if debug_print:
            done = False
            step = 0

            while not done:
                multi_actions = self.generate_action(act_space)
                _, _, trainable_done, _, _ = env.step(multi_actions)
                done = trainable_done["__all__"]
                step += 1
            total_timesteps += step
            print(f"\tSteps = {step}")
            time.time()
        t2 = time.time()

        profiler.stop()

        step_per_second = total_timesteps / (t2 - t1)
        print(f"Total Steps  {total_timesteps}")
        print(f"Steps/Second {step_per_second}")

        open_file_name = "testing.html"
        if hasattr(self, "experiment_config_id"):
            open_file_name = f"testing_{self.experiment_config_id}_{int(round(step_per_second))}.html"

        with open(open_file_name, "w", encoding="utf-8") as f_html:
            f_html.write(profiler.output_html())

        return step_per_second

    def generate_action(self, act_space):  # noqa: PLR6301
        """randomly select an action to take."""
        # generate a random action
        return {a_k: {s_k: s.sample() for s_k, s in a_s.spaces.items()} for a_k, a_s in act_space.spaces.items()}

    def get_callbacks(self) -> type[EnvironmentDefaultCallbacks]:  # noqa: PLR6301
        """Get the environment callbacks."""
        return EnvironmentDefaultCallbacks

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

    def _enable_episode_parameter_provider_checkpointing(self) -> None:
        base_trainer = self.config.tune_config["run_or_experiment"]

        trainer_class = get_trainable_cls(base_trainer)

        class EpisodeParameterProviderSavingTrainer(trainer_class):  # type: ignore[valid-type, misc]
            """
            Tune Trainer that adds capability to restore
            progress of the EpisodeParameterProvider on restoring training
            progress.
            """

            def save_checkpoint(self, checkpoint_path: str) -> None:
                """
                adds additional checkpoint saving functionality
                by also saving any episode parameter providers
                currently running.
                """
                tmp = super().save_checkpoint(checkpoint_path)

                checkpoint_folder = pathlib.Path(checkpoint_path)

                # Environment
                epp_name = ACT3MultiAgentEnv.episode_parameter_provider_name
                env = self.workers.local_worker().env
                epp: EpisodeParameterProvider = env.config.epp
                epp.save_checkpoint(checkpoint_folder / epp_name)

                # Agents
                for agent_name, agent_configs in env.agent_dict.items():
                    epp = agent_configs.config.epp
                    epp.save_checkpoint(checkpoint_folder / agent_name)

                return tmp

            def load_checkpoint(self, checkpoint_path: str) -> None:
                """
                adds additional checkpoint loading functionality
                by also loading any episode parameter providers
                with the checkpoint.
                """
                super().load_checkpoint(checkpoint_path)

                checkpoint_folder = pathlib.Path(checkpoint_path).parent

                # Environment
                epp_name = ACT3MultiAgentEnv.episode_parameter_provider_name
                env = self.workers.local_worker().env
                epp: EpisodeParameterProvider = env.config.epp
                epp.load_checkpoint(checkpoint_folder / epp_name)

                # Agents
                for agent_name, agent_configs in env.agent_dict.items():
                    epp = agent_configs.config.epp
                    epp.load_checkpoint(checkpoint_folder / agent_name)

        self.config.tune_config["run_or_experiment"] = EpisodeParameterProviderSavingTrainer

    def _add_trial_creator(self):
        """Updates the trial name based on the HPC Job Number and the trial name in the configuration."""
        if "trial_name_creator" not in self.config.tune_config:

            def trial_name_prefix(trial):
                """
                Args:
                    trial (Trial): A generated trial object.

                Returns
                -------
                trial_name (str): String representation of Trial prefixed
                    by the contents of the environment variable:
                    TRIAL_NAME_PREFIX
                    Or the prefix 'RUN' if none is set.
                """
                trial_prefix = os.environ.get("PBS_JOBID", os.environ.get("TRIAL_NAME_PREFIX", ""))
                trial_name = ""

                if "TrialName" in self.config.env_config:
                    trial_name = "-" + self.config.env_config["TrialName"] if trial_prefix else self.config.env_config["TrialName"]
                return f"{trial_prefix}{trial_name}-{trial}"

            self.config.tune_config["trial_name_creator"] = trial_name_prefix
