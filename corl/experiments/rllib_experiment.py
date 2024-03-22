"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import gc
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
from pydantic import AfterValidator, BaseModel, ConfigDict, ImportString, field_validator
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.rllib.policy.policy import Policy
from ray.tune.registry import get_trainable_cls, register_env

import corl.experiments.rllib_utils.wrappers as rllib_wrappers
from corl.environment.base_multi_agent_env import BaseCorlMultiAgentEnv
from corl.environment.default_env_rllib_callbacks import EnvironmentDefaultCallbacks
from corl.environment.environment_wrappers import GroupedAgentsEnv
from corl.environment.multi_agent_env import ACT3MultiAgentEnv
from corl.episode_parameter_providers import EpisodeParameterProvider
from corl.episode_parameter_providers.remote import RemoteEpisodeParameterProvider
from corl.experiments.base_experiment import BaseExperiment, BaseExperimentValidator, ExperimentFileParse
from corl.experiments.rllib_utils.policy_mapping_functions import PolicyIsAgent
from corl.libraries.factory import Factory
from corl.libraries.file_path import CorlDirectoryPath
from corl.parsers.yaml_loader import apply_patches, load_file
from corl.policies.base_policy import BasePolicyValidator


class RllibPolicyMappingValidator(BaseModel):
    """validator of data used to generate policy mapping"""

    functor: ImportString = PolicyIsAgent
    config: dict = {}


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

    ray_config: dict[str, typing.Any]
    env_config: dict[str, typing.Any]
    rllib_configs: dict[str, dict[str, typing.Any]]
    tune_config: dict[str, typing.Any]
    trainable_config: dict[str, typing.Any] | None = None
    hparam_search_class: ImportString | None = None
    hparam_search_config: dict[str, typing.Any] | None = None
    extra_callbacks: list[ImportString] | None = None
    extra_tune_callbacks: list[ImportString] | None = None
    trial_creator_function: ImportString | None = None
    policy_mapping: RllibPolicyMappingValidator = RllibPolicyMappingValidator()
    environment: str = "CorlMultiAgentEnv"
    policies: dict[str, str] = {}
    configuration_only: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("rllib_configs", mode="before")
    @classmethod
    def apply_patches_rllib_configs(cls, v):
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
                warnings.warn("having this validator apply patches is deprecated, please use the !merge yml directive", DeprecationWarning)
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
        typing.Dict[str, typing.Any]

        by patching the first dictionary in the list with each patch afterwards

        Returns:
            [type] -- [description]
        """
        if isinstance(v, list):
            warnings.warn("having this validator apply patches is deprecated, please use the !merge yml directive", DeprecationWarning)
            # OLD METHOD:
            #     env_config: [!include ../environments/core_capture_v1.yml, *env_config_updates]
            #
            # N.B. in this case the environment command line argument is not used.
            v = apply_patches(v)
        return v


def path_exists(v):
    """Check that the checkpoint path exists"""

    if v is None:
        return v

    sub_file_checks = []
    for filename in ["rllib_checkpoint.json", "policy_state.pkl"]:
        tmp = v / filename
        sub_file_checks.append(tmp.exists())
    if not all(sub_file_checks):
        raise RuntimeError(
            "Loading a checkpoint from the policy config 'checkpoint_path' requires "
            "specifying a rllib policy folder that has both a 'rllib_checkpoint.json' and a 'policy_state.pkl' in it."
        )
    return v


class RllibPolicyValidator(BasePolicyValidator):
    """
    policy_class: callable policy class None will use default from trainer
    checkpoint_path: if this policy should attempt to load weights from a checkpoint
    train: should this policy be trained
    Arguments:
        BaseModel: [description]

    Raises:
        RuntimeError: [description]

    Returns:
        [type] -- [description]
    """

    config: dict[str, typing.Any] = {}
    policy_class: ImportString | None = None
    checkpoint_path: typing.Annotated[CorlDirectoryPath, AfterValidator(path_exists)] | None = None
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
        self._register_envs()

    @classmethod
    def process_cli_args(cls, config: dict, cli_args: ExperimentFileParse):
        """Modifies/overrides config fields with command line arguments."""
        in_env_config = config["env_config"]

        # OLD METHOD:
        #     env_config: [!include ../environments/core_capture_v1.yml, *env_config_updates]
        #
        #   N.B. in this case the environment command line argument is not used.
        #
        # NEW METHOD
        #     env_config:
        #         default: [!include ../environments/core_capture_v1.yml, *env_config_updates]
        #         option1: [!include ../environments/core_capture_v1.yml, *env_config_updates]
        #         option2: [!include ../environments/core_capture_v1_mod.yml, *env_config_updates]
        if isinstance(in_env_config, dict):
            if cli_args.environment in in_env_config:
                config["env_config"] = config["env_config"][cli_args.environment]
            elif "simulator" in in_env_config and "episode_parameter_provider" in in_env_config:
                pass
            else:
                raise ValueError(
                    f"The selected env is not in the env_config dict --- Environment = {cli_args.environment} --- {in_env_config}"
                )

    @staticmethod
    def get_validator() -> type[RllibExperimentValidator]:
        return RllibExperimentValidator

    @property
    def get_policy_validator(self) -> type[RllibPolicyValidator]:
        """Return validator"""
        return RllibPolicyValidator

    def initialize_ray(self, compute_platform: str, debug: bool):
        """Initializes ray"""
        if compute_platform in ["ray"]:
            self._update_ray_config_for_ray_platform()

        if debug:
            self.config.ray_config["local_mode"] = True

        ray.init(**self.config.ray_config)

    def run_experiment(self, args: ExperimentFileParse) -> None:
        # Check for env key in env_config and if not present allow deprecated
        # path to still work --- else index and replace with selected env.
        rllib_config = self._select_rllib_config(args.compute_platform)

        self.initialize_ray(args.compute_platform, args.debug)

        self.config.env_config["agents"], self.config.env_config["agent_platforms"] = self.create_agents(
            args.platform_config, args.agent_config
        )

        if args.output:
            self.config.env_config["output_path"] = str(args.output)
            self.config.tune_config["storage_path"] = str(args.output)

        if args.name:
            self.config.env_config["TrialName"] = args.name

        self._add_trial_creator()

        if not self.config.ray_config["local_mode"]:
            self.config.env_config["episode_parameter_provider"] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                Factory(**self.config.env_config["episode_parameter_provider"]),
                actor_name=BaseCorlMultiAgentEnv.episode_parameter_provider_name,
            )

            for agent_name, agent_configs in self.config.env_config["agents"].items():
                agent_configs.class_config.config["episode_parameter_provider"] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                    Factory(**agent_configs.class_config.config["episode_parameter_provider"]), agent_name
                )
        tmp_env: ACT3MultiAgentEnv = self.create_environment()
        self.config.env_config["epp_registry"] = tmp_env.config.epp_registry
        policies, train_policies, available_policies = self.create_policies(tmp_env)

        self._update_rllib_config(rllib_config, train_policies, policies, args, available_policies)
        self._update_tune_config()

        self._enable_episode_parameter_provider_checkpointing()

        search_class = None
        if self.config.hparam_search_class is not None:
            if self.config.hparam_search_config is not None:
                search_class = self.config.hparam_search_class(**self.config.hparam_search_config)
            else:
                search_class = self.config.hparam_search_class()
            search_class.add_algorithm_hparams(rllib_config, self.config.tune_config)

        self.on_configuration_end(args, rllib_config)

        if self.config.configuration_only:
            return
        # this environment may have initialized something on worker process, so just be to sure it is not going to collide
        # with anything
        del tmp_env
        gc.collect()

        tune.run(
            config=rllib_config,
            # progress_reporter=(),
            **self.config.tune_config,
        )

    def create_environment(self) -> typing.Any:
        """Return environment from rllib environment creators"""
        return rllib_wrappers.get_rllib_environment_creator(self.config.environment)(self.config.env_config)

    def create_policies(self, environment):
        """Return various polices used in experiments"""
        env_policies = {
            agent_name: self.get_policy_validator(**agent_parse.policy_config)
            for agent_name, agent_parse in self.config.env_config["agents"].items()
        }
        experiment_policies = {name: self.get_policy_validator(**load_file(config)) for name, config in self.config.policies.items()}
        available_policies = experiment_policies | env_policies

        policies = {
            policy_name: (
                available_policies[policy_name].policy_class,
                policy_obs,
                environment.action_space[policy_name],
                available_policies[policy_name].config,
            )
            for policy_name, policy_obs in environment.observation_space.spaces.items()
            if available_policies[policy_name]
        }

        train_policies = [policy_name for policy_name in policies if available_policies[policy_name].train]
        return policies, train_policies, available_policies

    def on_configuration_end(self, args: ExperimentFileParse, rllib_config: dict[str, typing.Any]) -> None:
        """Subclass hook for additional configuration after the end of the standard configuration."""

    def _update_rllib_config(self, rllib_config, train_policies, policies, args: ExperimentFileParse, available_policies) -> None:
        """
        Update several rllib config fields
        tmp_ac: agent_classes from temporary environment to get obs/action space, these should be
                considered read only, and will be destroyed when training begins
        """

        rllib_config["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": self.config.policy_mapping.functor(**self.config.policy_mapping.config),
            "policies_to_train": train_policies,
        }

        rllib_config["env"] = self.config.environment
        callback_list: list[type[DefaultCallbacks]] = [self.get_callbacks()]
        if checkpoint_weight_files := {
            agent_name: agent_config.checkpoint_path
            for agent_name, agent_config in available_policies.items()
            if agent_config.checkpoint_path
        }:

            class WeightLoaderCallback(DefaultCallbacks):
                """
                callback that will load the preexisting weights
                into policies
                """

                def on_algorithm_init(self, *, algorithm, **kwargs):  # noqa: PLR6301
                    agent_weight_dict = {}
                    for agent_name, file_path in checkpoint_weight_files.items():
                        try:
                            git_repo = git.Repo(file_path, search_parent_directories=True)
                        except git.InvalidGitRepositoryError:
                            print(f"Checkpoint hash for {agent_name} at {file_path} is unknown")
                        else:
                            git_hash = git_repo.head.object.hexsha
                            print(f"Checkpoint hash for {agent_name} at {file_path}: {git_hash}")
                        restored = Policy.from_checkpoint(file_path)
                        agent_weight_dict[agent_name] = restored.get_weights()
                    algorithm.set_weights(agent_weight_dict)
                    algorithm.workers.sync_weights()

            callback_list.append(WeightLoaderCallback)
        if self.config.extra_callbacks:
            callback_list.extend(self.config.extra_callbacks)
        rllib_config["callbacks"] = make_multi_callbacks(callback_list)
        rllib_config["env_config"] = self.config.env_config
        now = datetime.now()
        rllib_config["env_config"]["output_date_string"] = f"{now.strftime('%Y%m%d_%H%M%S')}_{socket.gethostname()}"
        rllib_config["create_env_on_driver"] = True

        self._add_git_hashes_to_config(rllib_config)

        if args.debug:
            rllib_config["num_workers"] = 0

    def _update_tune_config(self) -> None:
        """
        Update tune config with extra callbacks
        """
        if self.config.extra_tune_callbacks:
            self.config.tune_config["callbacks"] = [callback() for callback in self.config.extra_tune_callbacks]

    def _add_git_hashes_to_config(self, rllib_config) -> None:
        """adds git hashes (or package version information if git information
        is unavailable) of key modules to rllib_config["env_config"]["git_hash"].
        Key modules are the following:
          - corl,
          - whatever cwd is set to at the time of the function call
            (notionally /opt/project / -rl-agents)
          - any other modules listed in rllib_config["env_config"]["plugin_paths"]

        This information is not actually used by BaseCorlMultiAgentEnv;
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
            rllib_config["env_config"]["git_hash"] = {}

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
            for item in rllib_config["env_config"]["plugin_paths"]:
                match0 = cp0.match(item)
                if match0 is None:
                    plugpath.append(item)

            plugpath.append("corl")

            # add git hashes to env_config dictionary
            for module0 in plugpath:
                env_hash_key = module0
                module1 = importlib.import_module(module0)
                modulefile = module1.__file__
                if modulefile is not None:
                    match0 = rp.match(modulefile)
                    if match0 is not None:
                        repo_path = match0.group("repopath")
                        try:
                            githash = git.Repo(repo_path, search_parent_directories=True).head.object.hexsha
                            rllib_config["env_config"]["git_hash"][env_hash_key] = githash
                            self._logger.info(f"{module0} hash: {githash}")
                        except git.InvalidGitRepositoryError:
                            # possibly installed in image but not a git repo
                            # look for version number
                            if hasattr(module1, "version") and hasattr(module1.version, "__version__"):
                                githash = module1.version.__version__
                                rllib_config["env_config"]["git_hash"][env_hash_key] = githash
                                self._logger.info(f"{module0} hash: {githash}")
                            else:
                                self._logger.warning(f"module: {module0}, repopath: {repo_path} is invalid git repo\n")
                                sys.stderr.write(f"module: {module0}, repopath: {repo_path} is invalid git repo\n")
        except ValueError:
            warnings.warn("Unable to add the github hash to experiment!!!")

    def get_callbacks(self) -> type[EnvironmentDefaultCallbacks]:  # noqa: PLR6301
        """Get the environment callbacks"""
        return EnvironmentDefaultCallbacks

    def _register_envs(self):  # noqa: PLR6301
        register_env("CorlMultiAgentEnv", lambda config: ACT3MultiAgentEnv(config))
        register_env("CorlGroupedAgentEnv", lambda config: GroupedAgentsEnv(config))

    def _select_rllib_config(self, platform: str | None) -> dict[str, typing.Any]:
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
        """Update the ray configuration for ray platforms"""
        self.config.ray_config["address"] = "auto"
        self.config.ray_config["log_to_driver"] = False

    def _enable_episode_parameter_provider_checkpointing(self) -> None:
        base_trainer = self.config.tune_config["run_or_experiment"]

        trainer_class = get_trainable_cls(base_trainer)

        def save_checkpoint(cls, checkpoint_path: str) -> None:
            """
            adds additional checkpoint saving functionality
            by also saving any episode parameter providers
            currently running
            """
            # mypy complains because the first argument to super isn't the current
            # `self` but this is actually OK since we in fact don't want super to
            # ascend the MRO of `self` but rather of `trainer_class`
            tmp = super(trainer_class, cls).save_checkpoint(checkpoint_path)  # type: ignore

            checkpoint_folder = pathlib.Path(checkpoint_path)

            # Environment
            epp_name = ACT3MultiAgentEnv.episode_parameter_provider_name
            env = cls.workers.local_worker().env
            epp: EpisodeParameterProvider = env.config.epp
            epp.save_checkpoint(checkpoint_folder / epp_name)

            # Agents
            for agent_name, agent_configs in env.agent_dict.items():
                epp = agent_configs.config.epp
                epp.save_checkpoint(checkpoint_folder / agent_name)

            return tmp

        def load_checkpoint(cls, checkpoint_path: str) -> None:
            """
            adds additional checkpoint loading functionality
            by also loading any episode parameter providers
            with the checkpoint
            """
            # mypy complains because the first argument to super isn't the current
            # `self` but this is actually OK since we in fact don't want super to
            # ascend the MRO of `self` but rather of `trainer_class`
            super(trainer_class, cls).load_checkpoint(checkpoint_path)  # type: ignore

            checkpoint_folder = pathlib.Path(checkpoint_path)

            # Environment
            epp_name = ACT3MultiAgentEnv.episode_parameter_provider_name
            env = cls.workers.local_worker().env
            epp: EpisodeParameterProvider = env.config.epp
            epp.load_checkpoint(checkpoint_folder / epp_name)

            # Agents
            for agent_name, agent_configs in env.agent_dict.items():
                epp = agent_configs.config.epp
                epp.load_checkpoint(checkpoint_folder / agent_name)

        # NOTE: this adds the EPP checkpointing while preserving the trainable name
        trainer_class.save_checkpoint = save_checkpoint
        trainer_class.load_checkpoint = load_checkpoint

        self.config.tune_config["run_or_experiment"] = trainer_class

    def _add_trial_creator(self):
        """Updates the trial name based on the HPC Job Number and the trial name in the configuration"""
        if "trial_name_creator" not in self.config.tune_config:
            if self.config.trial_creator_function is None:
                trial_prefix = os.environ.get("PBS_JOBID", os.environ.get("TRIAL_NAME_PREFIX", ""))
                trial_name = ""

                if "TrialName" in self.config.env_config:
                    trial_name = "-" + self.config.env_config["TrialName"] if trial_prefix else self.config.env_config["TrialName"]

                def trial_name_prefix(trial):
                    """
                    Args:
                        trial (Trial): A generated trial object.

                    Returns:
                        trial_name (str): String representation of Trial prefixed
                            by the contents of the environment variable:
                            PBS_JOBID or TRIAL_NAME_PREFIX
                            followed by the TrialName element of the environment configuration
                    """
                    return f"{trial_prefix}{trial_name}-{trial}"

                self.config.tune_config["trial_name_creator"] = trial_name_prefix
            else:
                self.config.tune_config["trial_name_creator"] = self.config.trial_creator_function
