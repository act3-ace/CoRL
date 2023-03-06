"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import os
import tempfile
import typing

import jsonargparse

from corl.evaluation.recording.folder import Folder
from corl.evaluation.runners.iterate_test_cases import IterateTestCases  # type: ignore
from corl.evaluation.runners.section_factories.engine.rllib.rllib_trainer import RllibTrainer
from corl.evaluation.runners.section_factories.plugins.plugins import Plugins
from corl.evaluation.runners.section_factories.task import Task
from corl.evaluation.runners.section_factories.teams import Teams
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import TestCaseManager

#################################################################################
# If you want to debug a configuration that already exists
# Remove comments from the following line and make sure `from_dict` is commented out
# Then run this file
#################################################################################
# from_file = ""
#################################################################################
# If you haven't make a yaml file for your setup can create it here
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
#                     class_path: corl.evaluation.loader.check_point_file.CheckpointFile
#                     init_args:
#                         checkpoint_filename: /opt/data/corl/agents/dummy_1v1/checkpoint-12
#                         policy_id: blue0
#         red:
#         -   platform_config: config/platforms/f16/sixdof_lcirst_rbr_point_mass.yml
#             agents:
#             -
#                 name: red0
#                 agent_config: config/tasks/1v1/agents/wld_slew_lcirst_delta.yml
#                 policy_config: config/policies/framestacking.yml
#                 agent_loader:
#                     class_path: corl.evaluation.loader.check_point_file.CheckpointFile
#                     init_args:
#                         checkpoint_filename: /opt/data/corl/agents/dummy_1v1/checkpoint-12
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
#             dir: /opt/data/corl/evaluation_1v1
#             append_timestamp: False
# """

######################################################
# ONLY MODIFY BELOW THIS LINE IF KNOW WHAT DOING
######################################################


def get_args(
    path: str = None,
    alternate_argv: typing.Optional[typing.Sequence[str]] = None
) -> typing.Tuple[typing.Type[jsonargparse.Namespace], typing.Type[jsonargparse.Namespace]]:
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
    parser.add_argument("--cfg", help="path to a json/yml file containing the running arguments", action=jsonargparse.ActionConfigFile)

    # Construct teams
    parser.add_class_arguments(Teams, "teams", instantiate=True)

    # How to construct the generic task/environment
    parser.add_class_arguments(Task, "task", instantiate=True)

    # How to generate the test cases to be provided to the environment
    # Different types of test case matrix representations will be added here

    # Define class which holds EPP info
    parser.add_class_arguments(TestCaseManager, "test_cases.test_case_manager", instantiate=True)

    # How to construct the plugins needed by generic evaluation
    parser.add_class_arguments(Plugins, "plugins", instantiate=True)

    # How to construct the engine to run evaluation
    # If more tasks types are added they will be appended here
    parser.add_class_arguments(
        RllibTrainer,
        "engine.rllib",
        instantiate=True,
    )

    # Recorders to publish results to
    parser.add_argument("recorders", type=typing.List[Folder])

    # Temporary directory
    parser.add_argument("tmpdir_base", type=str, default='/tmp')

    if path is None:
        args = parser.parse_args(args=alternate_argv)
    else:
        args = parser.parse_path(path)

    instantiate = parser.instantiate_classes(args)

    return instantiate, args


def main(instantiated_args: typing.Union[jsonargparse.Namespace, typing.Dict]):
    """Main function block to evaluate from a configuration

    Parameters:
    ----------
        instantiated_args: jsonargparse.Namespace
            Contains as a Namespace the instantiated objects needed to run evaluation.
    """

    if len(list(instantiated_args["test_cases"].values())) == 0:
        raise RuntimeError("No test_cases was provided, this is a non-op")
    if len(list(instantiated_args["test_cases"].values())) > 1:
        raise RuntimeError("Multiple test_cases were provided, this is a non-op")

    if len(list(instantiated_args["engine"].values())) == 0:
        raise RuntimeError("No engine was provided, this is a non-op")
    if len(list(instantiated_args["engine"].values())) > 1:
        raise RuntimeError("Multiple engines were provided, this is a non-op")

    os.makedirs(instantiated_args["tmpdir_base"], exist_ok=True)
    with tempfile.TemporaryDirectory(dir=instantiated_args["tmpdir_base"]) as tmpdir:
        evaluate = IterateTestCases(
            teams=instantiated_args["teams"],
            task=instantiated_args["task"],
            test_case_manager=instantiated_args["test_cases"]["test_case_manager"],
            plugins=instantiated_args["plugins"],
            engine=list(instantiated_args["engine"].values())[0],
            recorders=instantiated_args["recorders"],
            tmpdir=tmpdir
        )
        records = evaluate.run()

    return records


def pre_main(alternate_argv: typing.Optional[typing.Sequence[str]] = None):
    """
    calls gets current args and passes them to main
    """
    instantiated, _ = get_args(alternate_argv=alternate_argv)
    main(instantiated)


if __name__ == "__main__":
    pre_main()
