"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""


def update_rllib_experiment_for_test(experiment_class, experiment_file_validated, test_rllib_config, tmp_path):
    experiment_class.config.rllib_configs["local"].update(test_rllib_config)
    if "model" in experiment_class.config.rllib_configs["local"]:
        experiment_class.config.rllib_configs["local"]["model"].reset()

    experiment_class.config.ray_config["ignore_reinit_error"] = True
    if "_temp_dir" in experiment_class.config.ray_config:
        del experiment_class.config.ray_config["_temp_dir"]
    experiment_class.config.env_config["output_path"] = str(tmp_path / "training")
    experiment_class.config.tune_config["stop"]["training_iteration"] = 1
    experiment_class.config.tune_config["storage_path"] = str(tmp_path / "training")
    experiment_class.config.tune_config["checkpoint_freq"] = 1
    experiment_class.config.tune_config["max_failures"] = 1
    experiment_file_validated.compute_platform = "local"
