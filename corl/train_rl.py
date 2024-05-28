# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

import argparse
import logging
import pathlib
import typing
import warnings

import numpy as np
import yaml

from corl.experiments.base_experiment import BaseExperiment, ExperimentFileParse, ExperimentParse
from corl.libraries.config_file_watcher import LoggingSetup
from corl.parsers.yaml_loader import Loader, load_file

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_experiments_yml(config_filename):
    """
    parse experiment config
    """
    parsed_config = load_file(config_filename)
    if isinstance(parsed_config["config"], str):
        warnings.warn(
            """
                      Deprecation Warning corl.base_experiment:  The experiment config parameter is
                      defined as a file rather than a dictionary. Consider replacing as follows
                      config: file_name.yml -> config: !include file_name.yml
                      """
        )
        with open(config_filename, encoding="utf-8") as fp:
            config = yaml.load(fp, Loader)  # noqa: S506

        # replace task file with include
        include_str = f'!include { config["config"] }'
        config["config"] = include_str
        # write new yaml to string
        new_config = yaml.dump(config, default_style=None)
        # the add include string will have quotes and they must be removed
        new_config = new_config.replace(f"'{include_str}'", include_str)
        parsed_config = yaml.load(new_config, Loader)  # noqa: S506

    return parsed_config


def merge_cfg_and_args(cfg: dict, args: argparse.Namespace) -> dict:
    """
    override items in the cfg file with arguments from command line
    """
    if arg_val := args.compute_platform:
        cfg["compute_platform"] = arg_val
    elif "compute_platform" not in cfg:
        cfg["compute_platform"] = "default"

    for arg_name in ["environment", "name", "output"]:
        if arg_val := getattr(args, arg_name):
            cfg[arg_name] = arg_val
    if args.debug:
        cfg["debug"] = True
    return cfg


def build_experiment(args) -> tuple[BaseExperiment, ExperimentFileParse]:
    """
    handles the argparsing for CoRL, this function allows alternate arguments to be
    input to allow unit tests to use the same parsing code

    Args:
        alternate_argv (typing.Optional[typing.Sequence[str]], optional): _description_. Defaults to None.

    Returns:
        _type_: fully parsed arguments
    """
    cfg = parse_experiments_yml(config_filename=args.cfg)
    cfg = merge_cfg_and_args(cfg, args)
    experiment_file_validated = ExperimentFileParse(**cfg)
    experiment_parse = ExperimentParse(**experiment_file_validated.config)
    experiment_parse.experiment_class.process_cli_args(experiment_parse.config, experiment_file_validated)
    experiment_class = experiment_parse.experiment_class(**experiment_parse.config)

    return experiment_class, experiment_file_validated


def parse_corl_args(alternate_argv: typing.Sequence[str] | None = None) -> argparse.Namespace:
    """
    handles the argparsing for CoRL, this function allows alternate arguments to be
    input to allow unit tests to use the same parsing code

    Args:
        alternate_argv (typing.Optional[typing.Sequence[str]], optional): _description_. Defaults to None.

    Returns:
        _type_: fully parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=pathlib.Path,
        help="Path to config.yml file used to setup the training environment",
    )
    parser.add_argument(
        "--compute-platform",
        type=str,
        help="Compute platform [ace, hpc, local] of experiment. Used to select rllib_config. "
        "this will default to 'default' if not provided",
    )
    parser.add_argument(
        "--environment",
        type=str,
        help="Compute environment [env1, env2, default] of experiment. Used to select env_config",
    )
    parser.add_argument(
        "--name", action="store", help="Tells your specified experiment to update its name. Experiments may ignore this directive."
    )
    parser.add_argument(
        "--output",
        action="store",
        help="Tells your specified experiment to update its output directory.  Experiments may ignore this directive.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level. This will be overridden by --log-config if log config exists.",
    )
    parser.add_argument(
        "--log-config", type=pathlib.Path, default=pathlib.Path.cwd() / "logging.yml", help="Optional path to yaml logging configuration."
    )
    # Canonical way to set flags while support vscode `pickString`
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tells your specified experiment to switch configurations to debug mode.  Experiments may ignore this directive",
    )
    parser.set_defaults(debug=False)
    return parser.parse_args(args=alternate_argv)


def main() -> None:
    """
    Main method of the module that allows for arguments parsing  for experiment setup.
    """
    args = parse_corl_args()
    LoggingSetup(default_path=str(args.log_config), default_level=logging._nameToLevel[args.log_level])  # noqa: SLF001
    experiment_class, experiment_file_validated = build_experiment(args)
    experiment_class.run_experiment(experiment_file_validated)


if __name__ == "__main__":
    main()
