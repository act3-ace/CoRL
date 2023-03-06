"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
This Module contains logic to update the rllib config appropriately for evaluation purposes.
"""

import typing
import warnings

from pydantic import BaseModel

from corl.evaluation.runners.section_factories.plugins.config_updater import ConfigUpdate
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import TestCaseManager


class RllibConfigUpdateValidator(BaseModel):
    """
    Initialize the function to use to update the rllib config.

    Parameters:
    -----------
    output_dir : str
        output directory for output dump of intermediate materials
    pandas_test_cases: pd.DataFrame
        test_cases to use when overriding EPP
    workers: int
        number of workers
    envs_per_worker: int
        number of environments per worker
    horizon: int
        value of using horizon , 0 or 1
    """

    output_dir: typing.Optional[str] = None
    test_case_manager: typing.Optional[TestCaseManager] = None
    workers: int
    envs_per_worker: typing.Optional[int]
    horizon: typing.Optional[int]
    explore: bool = False

    class Config:  # pylint:disable=too-few-public-methods # Needed for pydantic
        """Config for pydantic
        """
        arbitrary_types_allowed = True


class RllibConfigUpdate(ConfigUpdate):
    """
    Initialize the function to use to update the rllib config.

    Parameters:
    -----------
    output_dir : typing.Optional[str]
        output directory for output dump of intermediate materials
    test_case_manager: typing.Optional[TestCaseManager]
        test_cases to use when overriding EPP
    workers: int
        number of workers
    envs_per_worker: int
        number of environments per worker
    horizon: int
        value of using horizon , 0 or 1
    """

    def __init__(self, **kwargs):
        self.config: RllibConfigUpdateValidator = self.get_validator(**kwargs)

    @property
    def get_validator(self) -> typing.Type[RllibConfigUpdateValidator]:
        """
        retrieve validator and appropriately setup the config

        Returns
        -------
        RllibConfigUpdateValidator -- validator used to generate a configuration"""
        return RllibConfigUpdateValidator

    def update(self, config: dict):
        config['create_env_on_driver'] = True
        config['num_workers'] = self.config.workers
        if self.config.envs_per_worker is not None:
            config['num_envs_per_worker'] = self.config.envs_per_worker
        config['num_cpus_for_driver'] = 1
        config['num_cpus_per_worker'] = 1
        config["num_gpus_per_worker"] = 0
        config["num_gpus"] = 0
        if self.config.output_dir:
            config['env_config']['output_path'] = str(self.config.output_dir)
        config['explore'] = self.config.explore
        config["batch_mode"] = "complete_episodes"

        # delegate EPP Override to TestCaseManager
        if self.config.test_case_manager is not None:
            config = self.config.test_case_manager.update_rllib_config(config)

        if self.config.horizon:
            if self.config.horizon < config['horizon']:
                warnings.warn('Done statistics might be invalid because EpisodeLengthDone will not be triggered.')
            config['horizon'] = self.config.horizon

        return config
