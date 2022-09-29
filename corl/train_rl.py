"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import pathlib
import typing
import warnings

import jsonargparse
import numpy as np

from corl.experiments.base_experiment import ExperimentParse
from corl.parsers.yaml_loader import load_file

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class MainUtilACT3Core:
    """
    Contains all the procedures that allow for argument parsing to setup an experiment
    """

    DEFAULT_CONFIG_PATH = str(pathlib.Path(__file__).parent.absolute() / 'config' / 'tasks' / 'single_lear_capture.yml')

    @staticmethod
    def parse_args(alternate_argv: typing.Optional[typing.Sequence[str]] = None):
        """
        Processes the arguments as main entry point for CoRL deep reinforcement training code

        Parameters
        ----------
        alternate_argv : Sequence[str], optional
            Arguments that should be parsed rather than sys.argv.  The default of None parses sys.argv.
            See https://docs.python.org/3/library/argparse.html#beyond-sys-argv.

        Returns
        -------
        namespace
            The arguments from the parser
        """

        parser = jsonargparse.ArgumentParser()
        parser.add_argument(
            "--cfg",
            help="an alternative way to provide arguments, the path to a json/yml file containing the running arguments",
            action=jsonargparse.ActionConfigFile
        )
        parser.add_argument(
            "--config",
            type=str,
            default=MainUtilACT3Core.DEFAULT_CONFIG_PATH,
            help=f"Path to config.yml file used to setup the training environment Default={MainUtilACT3Core.DEFAULT_CONFIG_PATH}",
        )

        parser.add_argument(
            "--compute-platform",
            type=str,
            default="auto",
            help="Compute platform [ace, hpc, local, auto] of experiment. Used to select rllib_config",
        )
        parser.add_argument(
            "-pc",
            "--platform-config",
            action="append",
            nargs=2,
            metavar=("platform-name", "platform-file"),
            help="the specification for a platform in the environment"
        )
        parser.add_argument(
            "-ac",
            "--agent-config",
            action="append",
            nargs=4,
            metavar=("agent-name", "platform-name", "configuration-file", "policy-file"),
            help="the specification for an agent in the environment"
        )
        parser.add_argument("-op", "--other-platform", action="append", nargs=2, metavar=("agent-name", "platform-file"), help="help:")
        parser.add_argument(
            '--debug',
            action='store_true',
            help="Tells your specified experiment to switch configurations to debug mode.  Experiments may ignore this flag"
        )

        parser.add_argument(
            '--name', action='store', help="Tells your specified experiment to update its name. Experiments may ignore this directive."
        )

        parser.add_argument(
            '--output',
            action='store',
            help="Tells your specified experiment to update its output directory.  Experiments may ignore this directive."
        )

        parser.add_argument('--profile', action='store_true', help="Tells experiment to switch configuration to profile mode")
        parser.add_argument('--profile-iterations', type=int, default=10)
        return parser.parse_args(args=alternate_argv)


def main():
    """
    Main method of the module that allows for arguments parsing  for experiment setup.
    """
    args = MainUtilACT3Core.parse_args()
    config = load_file(config_filename=args.config)

    # print(config)
    experiment_parse = ExperimentParse(**config)
    experiment_class = experiment_parse.experiment_class(**experiment_parse.config)
    print(experiment_class.config.dict())

    if experiment_parse.auto_system_detect_class is not None and args.compute_platform == 'auto':
        args.compute_platform = experiment_parse.auto_system_detect_class().autodetect_system()  # pylint: disable=not-callable
    elif experiment_parse.auto_system_detect_class is None and args.compute_platform == 'auto':
        args.compute_platform = "local"

    experiment_class.run_experiment(args)


if __name__ == "__main__":
    main()
