"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import argparse
import importlib
import logging
import os
import pathlib
import re
import socket
import sys
import typing
import warnings
from datetime import datetime

import git
import ray
from pydantic import PyObject, validator
from ray import tune
from ray.rllib.agents.callbacks import MultiCallbacks
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import get_trainable_cls

from corl.environment.default_env_rllib_callbacks import EnvironmentDefaultCallbacks
from corl.environment.multi_agent_env import ACT3MultiAgentEnv, ACT3MultiAgentEnvValidator
from corl.episode_parameter_providers import EpisodeParameterProvider
from corl.episode_parameter_providers.remote import RemoteEpisodeParameterProvider
from corl.experiments.base_experiment import BaseExperiment, BaseExperimentValidator
from corl.libraries.factory import Factory
from corl.libraries.rllib_setup_util import AutoRllibConfigSetup, auto_configure_rllib_config
from corl.parsers.yaml_loader import apply_patches
from corl.policies.base_policy import BasePolicyValidator


class RllibExperimentValidator(BaseExperimentValidator):
    """
    ray_config: dictionary to be fed into ray init, validated by ray init call
    env_config: environment configuration, validated by environment class
    rllib_configs: a mapping of compute platforms to rllib configs, see apply_patches_rllib_configs
                    for information on the typing
    tune_config: kwarg arguments to be sent to tune for this experiment
    extra_callbacks: extra rllib callbacks that will be added to the callback list
    trial_creator_function: this function will overwrite the default trial string creator
                            and allow more fine tune trial name creators
    """
    ray_config: typing.Dict[str, typing.Any]
    env_config: EnvContext
    rllib_configs: typing.Dict[str, typing.Dict[str, typing.Any]]
    tune_config: typing.Dict[str, typing.Any]
    trainable_config: typing.Optional[typing.Dict[str, typing.Any]]
    auto_rllib_config_setup: AutoRllibConfigSetup = AutoRllibConfigSetup()
    hparam_search_class: typing.Optional[PyObject]
    hparam_search_config: typing.Optional[typing.Dict[str, typing.Any]]
    extra_callbacks: typing.Optional[typing.List[PyObject]]
    trial_creator_function: typing.Optional[PyObject]

    @validator('rllib_configs', pre=True)
    def apply_patches_rllib_configs(cls, v):  # pylint: disable=no-self-argument, no-self-use
        """
        The dictionary of rllib configs may come in as a dictionary of
        lists of dictionaries, this function is responsible for collapsing
        the list down to a typing.Dict[str, typing.Dict[str, typing.Any]]
        instead of
        typing.Dict[str, typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]]]

        Raises:
            RuntimeError: [description]

        Returns:
            [type] -- [description]
        """
        if not isinstance(v, dict):
            raise RuntimeError("rllib_configs are expected to be a dict of keys to different compute configs")
        rllib_configs = {}
        for key, value in v.items():
            if isinstance(value, list):
                rllib_configs[key] = apply_patches(value)
            elif isinstance(value, dict):
                rllib_configs[key] = value
        return rllib_configs

    @validator('ray_config', 'tune_config', 'trainable_config', 'env_config', pre=True)
    def apply_patches_configs(cls, v):  # pylint: disable=no-self-argument, no-self-use
        """
        reduces a field from
        typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]]]
        to
        typing.Dict[str, typing.Any]

        by patching the first dictionary in the list with each patch afterwards

        Returns:
            [type] -- [description]
        """
        if isinstance(v, list):
            v = apply_patches(v)
        return v

    @validator('env_config')
    def no_horizon(cls, v):
        """Ensure that the horizon is not specified in the env_config."""
        if 'horizon' in v:
            raise ValueError('Cannot specify the horizon in the env_config')
        return v


class RllibPolicyValidator(BasePolicyValidator):
    """
    policy_class: callable policy class None will use default from trainer
    train: should this policy be trained
    Arguments:
        BaseModel {[type]} -- [description]

    Raises:
        RuntimeError: [description]

    Returns:
        [type] -- [description]
    """
    config: typing.Dict[str, typing.Any] = {}
    policy_class: typing.Union[PyObject, None] = None
    train: bool = True


class RllibExperiment(BaseExperiment):
    """
    The Rllib Experiment is an experiment for running
    multiagent configurable environments with patchable settings
    """

    def __init__(self, **kwargs) -> None:
        self.config: RllibExperimentValidator
        self._logger = logging.getLogger(RllibExperiment.__name__)
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[RllibExperimentValidator]:
        return RllibExperimentValidator

    @property
    def get_policy_validator(self) -> typing.Type[RllibPolicyValidator]:
        """Return validator"""
        return RllibPolicyValidator

    def run_experiment(self, args: argparse.Namespace) -> None:

        rllib_config = self._select_rllib_config(args.compute_platform)

        if args.compute_platform in ['ray']:
            self._update_ray_config_for_ray_platform()

        self._add_trial_creator()

        # This needs to be before the ray cluster is initialized
        if args.debug:
            self.config.ray_config['local_mode'] = True

        ray.init(**self.config.ray_config)

        ray_resources = ray.available_resources()

        auto_configure_rllib_config(rllib_config, self.config.auto_rllib_config_setup, ray_resources)

        self.config.env_config["agents"], self.config.env_config["agent_platforms"] = self.create_agents(
            args.platform_config, args.agent_config
        )

        self.config.env_config["horizon"] = rllib_config["horizon"]

        if args.output:
            self.config.env_config["output_path"] = args.output
            self.config.tune_config["local_dir"] = args.output

        if args.name:
            self.config.env_config["TrialName"] = args.name

        if args.other_platform:
            self.config.env_config["other_platforms"] = self.create_other_platforms(args.other_platform)

        if not self.config.ray_config['local_mode']:
            self.config.env_config['episode_parameter_provider'] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                Factory(**self.config.env_config['episode_parameter_provider']),
                actor_name=ACT3MultiAgentEnv.episode_parameter_provider_name
            )

            for agent_name, agent_configs in self.config.env_config['agents'].items():
                agent_configs.class_config.config['episode_parameter_provider'] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                    Factory(**agent_configs.class_config.config['episode_parameter_provider']), agent_name
                )

        self.config.env_config['epp_registry'] = ACT3MultiAgentEnvValidator(**self.config.env_config).epp_registry

        tmp = ACT3MultiAgentEnv(self.config.env_config)
        tmp_as = tmp.action_space
        tmp_os = tmp.observation_space
        tmp_ac = self.config.env_config['agents']

        policies = {
            policy_name: (
                tmp_ac[policy_name].policy_config["policy_class"],
                policy_obs,
                tmp_as[policy_name],
                tmp_ac[policy_name].policy_config["config"]
            )
            for policy_name,
            policy_obs in tmp_os.spaces.items()
            if tmp_ac[policy_name]
        }

        train_policies = [policy_name for policy_name in policies.keys() if tmp_ac[policy_name].policy_config["train"]]

        self._update_rllib_config(rllib_config, train_policies, policies, args)

        self._enable_episode_parameter_provider_checkpointing()

        if args.profile:
            if "stop" not in self.config.tune_config:
                self.config.tune_config["stop"] = {}
            self.config.tune_config["stop"]["training_iteration"] = args.profile_iterations

        search_class = None
        if self.config.hparam_search_class is not None:
            if self.config.hparam_search_config is not None:
                search_class = self.config.hparam_search_class(**self.config.hparam_search_config)
            else:
                search_class = self.config.hparam_search_class()
            search_class.add_algorithm_hparams(rllib_config, self.config.tune_config)

        tune.run(
            config=rllib_config,
            **self.config.tune_config,
        )

    def _update_rllib_config(self, rllib_config, train_policies, policies, args: argparse.Namespace) -> None:
        """
        Update several rllib config fields
        """

        rllib_config["multiagent"] = {
            "policies": policies, "policy_mapping_fn": lambda agent_id: agent_id, "policies_to_train": train_policies
        }

        rllib_config["env"] = ACT3MultiAgentEnv
        callback_list = [self.get_callbacks()]
        if self.config.extra_callbacks:
            callback_list.extend(self.config.extra_callbacks)  # type: ignore[arg-type]
        rllib_config["callbacks"] = MultiCallbacks(callback_list)
        rllib_config["env_config"] = self.config.env_config
        now = datetime.now()
        rllib_config["env_config"]["output_date_string"] = f"{now.strftime('%Y%m%d_%H%M%S')}_{socket.gethostname()}"
        rllib_config["create_env_on_driver"] = True
        rllib_config["batch_mode"] = "complete_episodes"

        self._add_git_hashes_to_config(rllib_config)

        if args.debug:
            rllib_config['num_workers'] = 0

    def _add_git_hashes_to_config(self, rllib_config) -> None:
        """adds git hashes (or package version information if git information
        is unavailable) of key modules to rllib_config["env_config"]["git_hash"].
        Key modules are the following:
          - corl,
          - whatever cwd is set to at the time of the function call
            (notionally /opt/project /)
          - any other modules listed in rllib_config["env_config"]["plugin_paths"]

        This information is not actually used by ACT3MultiAgentEnv;
        however, putting it in the env_config means that this
        information is saved to the params.pkl and thus is available
        for later inspection while seeking to understand the
        performance of a trained model.
        """
        try:
            # pattern used below to find root repository paths
            repo_pattern = r"(?P<repopath>.*)\/__init__.py"
            rp = re.compile(repo_pattern)

            corl_pattern = r"corl.*"
            cp0 = re.compile(corl_pattern)
            rllib_config["env_config"]["git_hash"] = dict()

            # store hash on cwd
            cwd = os.getcwd()
            try:
                githash = git.Repo(cwd, search_parent_directories=True).head.object.hexsha
                rllib_config["env_config"]["git_hash"]["cwd"] = githash
                self._logger.info(f"cwd hash: {githash}")
            except git.InvalidGitRepositoryError:
                self._logger.warning("cwd is not a git repo\n")

            # go ahead and strip out corl related things from plugin_path
            plugpath = []
            for item in rllib_config['env_config']['plugin_paths']:
                match0 = cp0.match(item)
                if match0 is None:
                    plugpath.append(item)

            plugpath.append('corl')

            # add git hashes to env_config dictionary
            for module0 in plugpath:
                env_hash_key = module0
                module1 = importlib.import_module(module0)
                modulefile = module1.__file__
                if modulefile is not None:
                    match0 = rp.match(modulefile)
                    if match0 is not None:
                        repo_path = match0.group('repopath')
                        try:
                            githash = git.Repo(repo_path, search_parent_directories=True).head.object.hexsha
                            rllib_config["env_config"]["git_hash"][env_hash_key] = githash
                            self._logger.info(f"{module0} hash: {githash}")
                        except git.InvalidGitRepositoryError:
                            # possibly installed in image but not a git repo
                            # look for version number
                            if hasattr(module1, 'version') and hasattr(module1.version, '__version__'):
                                githash = module1.version.__version__
                                rllib_config["env_config"]["git_hash"][env_hash_key] = githash
                                self._logger.info(f"{module0} hash: {githash}")
                            else:
                                self._logger.warning((f"module: {module0}, repopath: {repo_path}"
                                                      "is invalid git repo\n"))
                                sys.stderr.write((f"module: {module0}, repopath: {repo_path}"
                                                  "is invalid git repo\n"))
        except ValueError:
            warnings.warn("Unable to add the gitlab hash to experiment!!!")

    def get_callbacks(self) -> typing.Type[EnvironmentDefaultCallbacks]:
        """Get the environment callbacks"""
        return EnvironmentDefaultCallbacks

    def _select_rllib_config(self, platform: typing.Optional[str]) -> typing.Dict[str, typing.Any]:
        """Extract the rllib config for the proper computational platform

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
        """Update the ray configuration for ray platforms
        """
        self.config.ray_config['address'] = 'auto'
        self.config.ray_config['log_to_driver'] = False

    def _enable_episode_parameter_provider_checkpointing(self) -> None:

        base_trainer = self.config.tune_config["run_or_experiment"]

        trainer_class = get_trainable_cls(base_trainer)

        class EpisodeParameterProviderSavingTrainer(trainer_class):  # type: ignore[valid-type, misc]
            """
            Tune Trainer that adds capability to restore
            progress of the EpisodeParameterProvider on restoring training
            progress
            """

            def save_checkpoint(self, checkpoint_path):
                """
                adds additional checkpoint saving functionality
                by also saving any episode parameter providers
                currently running
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

            def load_checkpoint(self, checkpoint_path):
                """
                adds additional checkpoint loading functionality
                by also loading any episode parameter providers
                with the checkpoint
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
        """Updates the trial name based on the HPC Job Number and the trial name in the configuration
        """
        if "trial_name_creator" not in self.config.tune_config:
            if self.config.trial_creator_function is None:

                def trial_name_prefix(trial):
                    """
                    Args:
                        trial (Trial): A generated trial object.

                    Returns:
                        trial_name (str): String representation of Trial prefixed
                            by the contents of the environment variable:
                            TRIAL_NAME_PREFIX
                            Or the prefix 'RUN' if none is set.
                    """
                    trial_prefix = os.environ.get('PBS_JOBID', os.environ.get('TRIAL_NAME_PREFIX', ""))
                    trial_name = ""

                    if "TrialName" in self.config.env_config.keys():
                        if trial_prefix:
                            trial_name = "-" + self.config.env_config["TrialName"]
                        else:
                            trial_name = self.config.env_config["TrialName"]
                    return f"{trial_prefix}{trial_name}-{trial}"

                self.config.tune_config["trial_name_creator"] = trial_name_prefix
            else:
                self.config.tune_config["trial_name_creator"] = self.config.trial_creator_function
