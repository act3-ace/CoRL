"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import jsonargparse

from corl.evaluation.connection.base_eval_connection import ConnectionValidator, EvalConfig
from corl.evaluation.launchers.evaluate_runner import EvalRunner
from corl.evaluation.runners.section_factories.engine.rllib.rllib_trainer import ray_context
from corl.libraries.config_file_watcher import LoggingSetup
from corl.libraries.context import add_context
from corl.parsers.yaml_loader import load_file

#################################################################################
# If you want to debug a configuration that already exists
# Remove comments from the following line and make sure `from_dict` is commented out
# Then run this file
#################################################################################
# from_file = ""
#################################################################################
# If you haven't made a yaml file for your setup, you can create it here
# Remove comments from the following definition and make sure `from_file` is commented out
#################################################################################

# from_yml = """
# teams:
#     team_participant_map:
#         blue:
#         -   platform_config: config/platforms/f16/sixdof_lcirst_rbr_point_mass.yml
#             agents:
#             -
#                 name: blue0
#                 agent_config: config/tasks/1v1/agents/wld_slew_lcirst_delta.yml
#                 policy_config: config/policies/framestacking.yml
#                 agent_loader:
#                     class_path: corl.evaluation.loader.policy_checkpoint.policy_checkpoint
#                     init_args:
#                         checkpoint_filename: /tmp/data/corl/agents/dummy_1v1/checkpoint-12
#                         policy_id: blue0
#         red:
#         -   platform_config: config/platforms/f16/sixdof_lcirst_rbr_point_mass.yml
#             agents:
#             -
#                 name: red0
#                 agent_config: config/tasks/1v1/agents/wld_slew_lcirst_delta.yml
#                 policy_config: config/policies/framestacking.yml
#                 agent_loader:
#                     class_path: corl.evaluation.loader.policy_checkpoint.PolicyCheckpoint
#                     init_args:
#                         checkpoint_filename: /tmp/data/corl/agents/dummy_1v1/checkpoint-12
#                         policy_id: blue0

#     # If platform names ever change from <side><idx> format this will need to be changed
#     participant_id_schema: "%team_name%%idx%"
# task:
#   TaskGenerator:
#     task_config_file: config/tasks/1v1/tasks/multi_agent_large_wld.yml
# test_cases:
#   pandas:
#     data: config/evaluation/test_cases/1v1_f16_16.yml
#     source_form: FILE_YAML_CONFIGURATION
#     randomize: False
# recorders:
#     -
#         class_path: corl.evaluation.recording.Folder
#         init_args:
#             dir: /tmp/data/corl/evaluation_1v1
#             append_timestamp: False
# """

######################################################
# ONLY MODIFY BELOW THIS LINE IF KNOW WHAT DOING
######################################################


def get_args(path: str | None = None, alternate_argv: Sequence[str] | None = None) -> tuple[jsonargparse.Namespace, jsonargparse.Namespace]:
    """
    This is a method that allows for collection of arguments for evaluation

    Parameters:
    ----------
    path: str
        An optional argument to a path to config file that needs to be parsed to collect evaluation run arguments.

    Returns:

    instantiated: jsonargparse.Namespace
        A jsonargparse namespace containing instantiated objects to use for evaluation.
    args: jsonargparse.Namespace
        A jsonargparse namespace containing the basic run arguments to use for evaluaion. Used by the pipeline.

    """

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, help="path to a json/yml file containing the running arguments")
    # parser.add_argument("--connection_cfg", type=Path, help="path to a json/yml file containing the connection arguments")

    # Temporary directory
    parser.add_argument("tmpdir_base", type=Path, default="/tmp")  # noqa: S108

    parser.add_argument("--include_dashboard", action="store_true")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level. This will be overridden by --log-config if log config exists.",
    )

    parser.add_argument("--log-config", type=Path, default=Path.cwd() / "logging.yml", help="Optional path to yaml logging configuration.")

    args = parser.parse_args(args=alternate_argv) if path is None else parser.parse_path(path)

    instantiated = parser.instantiate_classes(args)

    return instantiated, args


def load_config(config_file: Path) -> dict[str, Any]:
    """Loads config from file"""
    return cast(dict[str, Any], load_file(config_filename=config_file))


def main(instantiated_args: jsonargparse.Namespace, config: dict[str, Any]):
    """Main function block to evaluate from a configuration

    Parameters:
    ----------
        instantiated_args: jsonargparse.Namespace
            Contains as a Namespace the instantiated objects needed to run evaluation.
    """
    if "experiment" not in config:
        config = {"experiment": config}

    connection = ConnectionValidator(**config).connection

    with add_context({"connection": connection}):
        kwargs = {"path": instantiated_args.cfg, "raw_config": config, **config}
        eval_schema = EvalConfig(**kwargs)

        with ray_context(local_mode=eval_schema.experiment.engine.rllib.debug_mode, include_dashboard=instantiated_args.include_dashboard):
            eval_runner = EvalRunner(tmpdir_base=instantiated_args.tmpdir_base, **{**config, "connection": connection})
            eval_runner(eval_schema)
            eval_runner.run()


def pre_main(alternate_argv: Sequence[str] | None = None):
    """
    calls gets current args and passes them to main
    """
    instantiated, args = get_args(alternate_argv=alternate_argv)
    LoggingSetup(default_path=str(args.log_config), default_level=logging._nameToLevel[args.log_level])  # noqa: SLF001
    cfg = load_config(instantiated.cfg)
    main(instantiated, cfg)


if __name__ == "__main__":
    pre_main()
