"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Configures a rllib trainer to run evaluation
"""
import copy
import dataclasses
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import ray
from pydantic import AfterValidator, BaseModel, Field, ImportString
from pydantic.dataclasses import dataclass
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.rllib.utils.typing import EnvType
from ray.tune.registry import get_trainable_cls

from corl.environment.multi_agent_env import ACT3MultiAgentEnv, ACT3MultiAgentEnvValidator
from corl.episode_parameter_providers.remote import RemoteEpisodeParameterProvider
from corl.evaluation.eval_logger_name import EVAL_LOGGER_NAME
from corl.evaluation.runners.section_factories.engine.rllib.default_evaluation_callbacks import DefaultEvaluationCallbacks
from corl.evaluation.runners.section_factories.engine.rllib.eval_config_updates import RllibConfigUpdate
from corl.evaluation.runners.section_factories.engine.rllib.interruptable_callback import InterruptableCallback
from corl.evaluation.runners.section_factories.plugins.environment_state_extractor import EnvironmentStateExtractor
from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer
from corl.evaluation.runners.section_factories.plugins.plugins import Plugins
from corl.evaluation.runners.section_factories.task import Experiment
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import TestCaseStrategy
from corl.libraries.algorithm_helper import reset_epp
from corl.libraries.factory import Factory


@contextmanager
def ray_context(*args, **kwargs):
    """Context manager to initialize and shutdown ray if not already initialized

    This function does not reinitialized an existing ray.  It takes any parameters accepted by
    ray.init() and passes them directly to ray.init().
    """
    if not ray.is_initialized():
        print("+" * 100)
        print("Initialize the ray context - ray.init")
        print("+" * 100)
        ray.init(*args, **kwargs)
        try:
            yield
        finally:
            print("+" * 100)
            print("Shutdown the ray context - ray.shutdown")
            print("+" * 100)
            ray.shutdown()
    else:
        yield


def callback_factory(serializer: PlatformSerializer, environment_state_extractor: EnvironmentStateExtractor | None = None):
    """Factory method to inject the platform serilization object

    Returns:
        _type_: Class that implements DefaultEvaluationCallbacks, but with default constructor
    """

    class ConcreteCallbacks(DefaultEvaluationCallbacks):
        """Class factory

        Args:
            DefaultEvaluationCallbacks (_type_): _description_
        """

        def platform_serializer(self):  # noqa: PLR6301
            """
            Method specifying the serialize function
            """
            return serializer

        def extract_environment_state(self, env_state):  # noqa: PLR6301
            if environment_state_extractor is None:
                return {}
            return copy.deepcopy(environment_state_extractor.extract(env_state))

    return ConcreteCallbacks


def check_subclass(v):
    if not issubclass(v, DefaultCallbacks):
        raise ValueError(f"Invalid type: {v}, must subclass {DefaultCallbacks}")
    return v


@dataclass
class RllibTrainer:
    """Configuration and logic to generate a Rllib trainer instance"""

    # Evaluation setup items
    callbacks: list[Annotated[ImportString, AfterValidator(check_subclass)]] = Field(default_factory=list)

    # Evaluation Core
    trainer_cls: str | type[Algorithm] = dataclasses.field(default="PPO")
    debug_mode: bool = dataclasses.field(default=False)
    workers: int = dataclasses.field(default=0)
    envs_per_worker: int | None = dataclasses.field(default=None)
    horizon: int | None = dataclasses.field(default=None)
    explore: bool = dataclasses.field(default=False)
    output_dir: Path | None = None

    env_cls: type[ACT3MultiAgentEnv] = ACT3MultiAgentEnv
    env_validator_cls: type[ACT3MultiAgentEnvValidator] = ACT3MultiAgentEnvValidator

    def generate(
        self,
        experiment: Experiment,
        test_case_manager: TestCaseStrategy | None = None,
        plugins: Plugins | None = None,
    ) -> Algorithm:
        """Generate the algorithm

        Arguments:
            experiment {Experiment} -- Experiment to load
            output_dir {str} -- directory to output to
            pandas_test_cases {pd.DataFrame} -- test cases

        Returns:
            Algorithm -- Resulting algorithm object
        """

        logger = logging.getLogger(EVAL_LOGGER_NAME)

        #############################################
        ## Manipulate the rllib_config for evaluation
        assert experiment.rllib_config is not None
        rllib_config = copy.deepcopy(experiment.rllib_config)
        rllib_config["env_config"] = experiment.env_config
        rllib_config["env"] = self.env_cls

        # default config manipulation
        default_eval_config_update = RllibConfigUpdate(
            output_dir=self.output_dir,
            test_case_manager=test_case_manager,
            workers=self.workers,
            envs_per_worker=self.envs_per_worker,
            explore=self.explore,
        )
        rllib_config = default_eval_config_update.update(rllib_config)
        if self.horizon is not None:
            rllib_config["horizon"] = self.horizon
            rllib_config["env_config"]["horizon"] = self.horizon
            rllib_config["train_batch_size"] = self.horizon
            rllib_config["sgd_minibatch_size"] = min(1000, int(self.horizon / 10))
            rllib_config["rollout_fragment_length"] = self.horizon

        # Any other config manipulation
        if plugins is not None:
            for cfg_update in plugins.eval_config_update:
                rllib_config = cfg_update.update(rllib_config)

        #############################################
        ## Setup environment epps
        if self.workers > 0:
            # raise NotImplementedError("worker > 0 has not been implemented/tested")
            logger.info("Performing distributed rollout with %s workers", self.workers)

            rllib_config["env_config"]["episode_parameter_provider"] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                Factory(**rllib_config["env_config"]["episode_parameter_provider"], namespace="evaluation"),
                actor_name=ACT3MultiAgentEnv.episode_parameter_provider_name,
            )

            for agent_name, agent_configs in rllib_config["env_config"]["agents"].items():
                agent_configs.class_config.config["episode_parameter_provider"] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                    Factory(**agent_configs.class_config.config["episode_parameter_provider"], namespace="evaluation"), agent_name
                )
        else:
            logger.info("Performing serial rollout")

        # This constructs all the EPPs once and uses them to configure all the environments with
        rllib_config["env_config"]["epp_registry"] = self.env_validator_cls(**rllib_config["env_config"]).epp_registry

        #############################################
        ## Setup environment callbacks
        callbacks_list: list[type[DefaultCallbacks]] = [InterruptableCallback]

        # Add the default platform serialization callback
        if plugins is not None:
            callbacks_list.append(callback_factory(plugins.platform_serialization, plugins.environment_state_extractor))

        # add all other callbacks given to this module
        callbacks_list.extend(iter(self.callbacks))
        rllib_config["callbacks"] = make_multi_callbacks(callbacks_list)
        rllib_config["disable_env_checking"] = True

        #############################################
        ## Get the algorithm type
        trainable_cls: type[Algorithm]
        if isinstance(self.trainer_cls, str):
            trainable_cls = get_trainable_cls(self.trainer_cls)
        elif issubclass(self.trainer_cls, Algorithm):
            trainable_cls = self.trainer_cls
        else:
            raise RuntimeError('"trainer_cls" field must either be a string or a `Algorithm` type')

        #############################################
        ## Wrap the algorithm type with one that will kill any remote epps on cleanup
        class AlgorithmWithProperCleanup(trainable_cls):  # type: ignore
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def cleanup(self):
                def shutdown_simulator(env: EnvType):
                    if isinstance(env, ACT3MultiAgentEnv):
                        env.simulator.shutdown()

                def shutdown_epps(env: EnvType):
                    if isinstance(env, ACT3MultiAgentEnv):
                        for epp in env.config.epp_registry.values():
                            epp.shutdown()

                def cleanup_env(env: EnvType) -> list:
                    shutdown_simulator(env)
                    shutdown_epps(env)
                    return []

                if self.workers is not None:
                    self.workers.foreach_env(cleanup_env)

                super().cleanup()

        algorithm = AlgorithmWithProperCleanup(config=rllib_config, env=self.env_cls)

        #############################################
        ## Reset the epps as they may be in a non-default state after Algorithm construction
        reset_epp(algorithm)

        return algorithm


class RllibConfig(BaseModel):
    """Wrapper for RllibTrainer"""

    rllib: RllibTrainer
