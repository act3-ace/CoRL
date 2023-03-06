"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import typing
import warnings

import jsonargparse
import numpy as np

from corl.experiments.base_experiment import ExperimentParse
from corl.parsers.agent_and_platform import CorlAgentsConfigArgs, CorlPlatformsConfigArgs
from corl.parsers.yaml_loader import load_file

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_corl_args(alternate_argv: typing.Optional[typing.Sequence[str]] = None):
    """
    handles the argparsing for CoRL, this function allows alternate arguments to be
    input to allow unit tests to use the same parsing code

    Args:
        alternate_argv (typing.Optional[typing.Sequence[str]], optional): _description_. Defaults to None.

    Returns:
        _type_: fully parsed arguments
    """
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--cfg", help="path to a json/yml file containing the running arguments", action=jsonargparse.ActionConfigFile)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yml file used to setup the training environment",
    )
    # Construct platforms and agents
    parser.add_class_arguments(CorlPlatformsConfigArgs, instantiate=True)
    parser.add_class_arguments(CorlAgentsConfigArgs, instantiate=True)
    parser.add_argument(
        "--compute-platform",
        type=str,
        default="auto",
        help="Compute platform [ace, hpc, local, auto] of experiment. Used to select rllib_config",
    )
    parser.add_argument(
        '--name', action='store', help="Tells your specified experiment to update its name. Experiments may ignore this directive."
    )
    parser.add_argument(
        '--output',
        action='store',
        help="Tells your specified experiment to update its output directory.  Experiments may ignore this directive."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Tells your specified experiment to switch configurations to debug mode.  Experiments may ignore this flag"
    )
    return parser.parse_args(args=alternate_argv)


def main():
    """
    Main method of the module that allows for arguments parsing  for experiment setup.
    """

    args = parse_corl_args()

    config = load_file(config_filename=args.config)
    experiment_parse = ExperimentParse(**config)

    experiment_parse.experiment_class.process_cli_args(experiment_parse.config, args)

    experiment_class = experiment_parse.experiment_class(**experiment_parse.config)

    if experiment_parse.auto_system_detect_class is not None and args.compute_platform == 'auto':
        args.compute_platform = experiment_parse.auto_system_detect_class().autodetect_system()  # pylint: disable=not-callable
    elif experiment_parse.auto_system_detect_class is None and args.compute_platform == 'auto':
        args.compute_platform = "local"

    experiment_class.run_experiment(args)


if __name__ == "__main__":
    main()
