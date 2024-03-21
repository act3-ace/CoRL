"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Entry / Launch point to run the storage process.
"""

import typing

import jsonargparse

from corl.evaluation.util.storage import Storage


def get_args(path: str | None = None) -> jsonargparse.Namespace:
    """
    Obtain running arguments for the storage process.

    Parameters
    ----------
    path : str
        optional path to a yaml config containing running arguments

    Returns
    -------
    instantiated: jsonargparse.Namespace
        namespace object that contains parsed running arguments
    """

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--cfg", action=jsonargparse.ActionConfigFile, help="the means by which to specify a yaml config to storage")
    parser.add_argument("--storage_utility", type=list[Storage], help="storage class to use")
    parser.add_argument(
        "--artifacts_location_config",
        type=dict[str, typing.Any],
        help="where are the eval_data_location, metric_file_location, event_table_location, agent_checkpoints ",
    )
    args = parser.parse_path(path) if path else parser.parse_args()
    return parser.instantiate_classes(args)


def main(instantiated_args):
    """
    Core logic for the storage process.

    Parameters
    ----------
    instantiated_args: jsonargparse.Namespace
        contains running arguments for the storage process.
    """
    for storage in instantiated_args["storage_utility"]:
        storage.load_artifacts_location(instantiated_args["artifacts_location_config"])
        storage.store()


def pre_main():
    """
    calls gets current args and passes them to main
    """
    instantiated, _ = get_args()
    main(instantiated)


if __name__ == "__main__":
    pre_main()
