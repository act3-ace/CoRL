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
import typing
from contextlib import contextmanager

import ray
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.tune.registry import get_trainable_cls

from corl.environment.multi_agent_env import ACT3MultiAgentEnv, ACT3MultiAgentEnvValidator
from corl.episode_parameter_providers.remote import RemoteEpisodeParameterProvider
from corl.evaluation.eval_logger_name import EVAL_LOGGER_NAME
from corl.evaluation.runners.section_factories.engine.rllib.default_evaluation_callbacks import DefaultEvaluationCallbacks
from corl.evaluation.runners.section_factories.engine.rllib.eval_config_updates import RllibConfigUpdate
from corl.evaluation.runners.section_factories.plugins.environment_state_extractor import EnvironmentStateExtractor
from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer
from corl.evaluation.runners.section_factories.plugins.plugins import Plugins
from corl.evaluation.runners.section_factories.task import Experiment
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import TestCaseManager
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


def callback_factory(serializer: PlatformSerializer, environment_state_extractor: typing.Optional[EnvironmentStateExtractor] = None):
    """Factory method to inject the platform serilization object

    Returns:
        _type_: Class that impliments DefaultEvaluationCallbacks, but with default constructor
    """

    class ConcreteCallbacks(DefaultEvaluationCallbacks):
        """Class factory

        Args:
            DefaultEvaluationCallbacks (_type_): _description_
        """

        def platform_serializer(self):
            """
            Method specifying the serialize function
            """
            return serializer

        def extract_environment_state(self, env_state):
            if environment_state_extractor is None:
                return {}
            return copy.deepcopy(environment_state_extractor.extract(env_state))

    return ConcreteCallbacks


@dataclasses.dataclass
class RllibTrainer:
    """Configuration and logic to generate a Rllib trainer instance
    """
    # Evaluation setup items
    callbacks: typing.List[typing.Type[DefaultCallbacks]]

    # Evaluation Core
    trainer_cls: typing.Union[str, typing.Type[Algorithm]] = dataclasses.field(default="PPO")
    debug_mode: bool = dataclasses.field(default=False)
    workers: int = dataclasses.field(default=0)
    envs_per_worker: typing.Optional[int] = dataclasses.field(default=None)
    horizon: typing.Optional[int] = dataclasses.field(default=None)
    explore: bool = dataclasses.field(default=False)

    env_cls: typing.Type[ACT3MultiAgentEnv] = ACT3MultiAgentEnv
    env_validator_cls: typing.Type[ACT3MultiAgentEnvValidator] = ACT3MultiAgentEnvValidator

    def generate(
        self,
        experiment: Experiment,
        output_dir: typing.Optional[str] = None,
        test_case_manager: typing.Optional[TestCaseManager] = None,
        plugins: typing.Optional[Plugins] = None
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

        rllib_config = copy.deepcopy(experiment.rllib_config)
        rllib_config['env_config'] = experiment.env_config
        rllib_config["env"] = self.env_cls

        # default config manipulation
        default_eval_config_update = RllibConfigUpdate(
            output_dir=output_dir,
            test_case_manager=test_case_manager,
            workers=self.workers,
            envs_per_worker=self.envs_per_worker,
            horizon=self.horizon,
            explore=self.explore
        )
        rllib_config = default_eval_config_update.update(rllib_config)

        # Any other config manipulation
        if plugins is not None:
            for cfg_update in plugins.eval_config_update:
                rllib_config = cfg_update.update(rllib_config)

        #############################################
        ## Setup environment callbacks

        if self.workers > 0:
            # raise NotImplementedError("worker > 0 has not been implimented/tested")
            logger.info('Performing distributed rollout with %s workers', self.workers)

            rllib_config['env_config']['episode_parameter_provider'] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                Factory(**rllib_config['env_config']['episode_parameter_provider'], namespace='evaluation'),
                actor_name=ACT3MultiAgentEnv.episode_parameter_provider_name
            )

            for agent_name, agent_configs in rllib_config['env_config']['agents'].items():
                agent_configs.class_config.config['episode_parameter_provider'] = RemoteEpisodeParameterProvider.wrap_epp_factory(
                    Factory(**agent_configs.class_config.config['episode_parameter_provider'], namespace='evaluation'), agent_name
                )
        else:
            logger.info('Performing serial rollout')

        rllib_config['env_config']['epp_registry'] = self.env_validator_cls(**rllib_config['env_config']).epp_registry

        # Callbacks setup
        callbacks_list = []

        # Add the default platform serialization callback
        if plugins is not None:
            callbacks_list.append(callback_factory(plugins.platform_serialization, plugins.environment_state_extractor))

        # add all other callbacks given to this module
        for callback in self.callbacks:
            callbacks_list.append(callback)

        rllib_config['callbacks'] = MultiCallbacks(callbacks_list)
        rllib_config["disable_env_checking"] = True

        #############################################
        ## Create the algorithm
        trainable_cls: typing.Type[Algorithm]
        if isinstance(self.trainer_cls, str):
            trainable_cls = get_trainable_cls(self.trainer_cls)
        elif isinstance(self.trainer_cls, Algorithm):
            trainable_cls = self.trainer_cls
        else:
            raise RuntimeError("\"trainer_cls\" field must either be a string or a `Algorithm` type")

        training_instance = trainable_cls(config=rllib_config, env=self.env_cls)
        return training_instance
