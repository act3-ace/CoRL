# type: ignore[union-attr]
"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Contains functionality to perform evaluation by iterating over given test cases
"""
import dataclasses
import logging
import typing

import numpy as np
from ray.rllib.policy.sample_batch import concat_samples
from ray.util.debug import disable_log_once_globally

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.eval_logger_name import EVAL_LOGGER_NAME
from corl.evaluation.evaluation_outcome import EvaluationOutcome
from corl.evaluation.recording.i_recorder import IRecord, IRecorder
from corl.evaluation.runners.section_factories.engine.rllib.rllib_trainer import RllibTrainer, ray_context
from corl.evaluation.runners.section_factories.plugins.plugins import Plugins
from corl.evaluation.runners.section_factories.task import Experiment, Task
from corl.evaluation.runners.section_factories.teams import Agent, Teams
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import TestCaseManager


def process_single_episode(worker):
    """
    Args:
        worker (_type_): _description_

    Returns:
        _type_: _description_
    """
    batches = [worker.input_reader.next()]
    batch = concat_samples(batches)
    worker.callbacks.on_sample_end(worker=worker, samples=batch)
    worker.output_writer.write(batch)
    return batch


@dataclasses.dataclass
class IterateTestCases:
    """Perform an evaluation by iterating over a set of given test cases
    """
    teams: Teams
    task: Task
    test_case_manager: TestCaseManager
    recorders: typing.List[IRecorder]
    plugins: Plugins
    tmpdir: str

    engine: RllibTrainer

    def run(self) -> typing.List[IRecord]:  # pylint: disable=too-many-locals,too-many-statements
        """Run evaluation

        Returns:
            typing.List[IRecord] -- List of records generated after evaluation finished

        Raises:
            RuntimeError -- If the number of consecutive test that do an invalid test exceeds threshold.
        """

        logger = logging.getLogger(EVAL_LOGGER_NAME)

        # Disable logging because can cause slowdowns
        disable_log_once_globally()

        # # generate and configure the output directory
        # output_dir =  self.output_folder.resolve()
        experiment = Experiment(task=self.task, teams=self.teams)

        # The following context must be valid during the running of parallel workers and saving of data
        # Important that this stay here.
        with ray_context(local_mode=self.engine.debug_mode, include_dashboard=False):
            # generate the algorithm
            algorithm = self.engine.generate(experiment, self.tmpdir, self.test_case_manager, self.plugins)  # type: ignore

            # Now insert each participant's weights into the algorithm
            def apply(agent: Agent):  # pylint:disable=unused-argument
                agent.agent_loader.apply_to_algorithm(algorithm, agent.name)

            self.teams.iterate_on_participant(apply)
            algorithm.workers.sync_weights()  # type: ignore

            # Do evaluation using ParallelRollouts
            # See Issue https://git.aoc-pathfinder.cloud/aaco/ai/act3-rllib-agents/-/issues/1423 for more discussion on this block
            # ParallelRollouts is given a set of test_cases for it to do
            # It will do those test cases, however not all test cases will take the same amount of time
            #
            # When all test cases have been completed a worker will "rollover" and start re-executing previous test case
            # when such a rollover is done in a workers batch it will have a ['eval'[0]['test_case'] value of None
            # This is referred to as an "invalid" test
            # See tabular_parameter_provider.py for rollover logic
            # Setting of \"test_case\" field is done in the EnvironmentCallback
            #
            # Since the rollout will rollover if we don't stop it then it will sample forever
            # The challenge becomes how do we know when all test cases have been processed and we can stop sampling the Parallel rollouts
            # This approach accomplishes this through two maps:
            #  - test_status_scheduled : A map of the test_cases index that have yet to be completed
            #  - test_status_completed : A map of the test_cases that have been completed
            # When a test is done it will be moved from scheduled to completed
            # We know we are done sampling when test_status_scheduled map is empty
            #
            # All that said the while(True) loop this block is based on is scary
            # There is a fear that if a test_case breaks for whatever reason workers
            # will go off re-executing tests for forever.
            # To address this if after a rollout no test are executed we increment a counter.
            # When that counter reaches a threshold then we raise an exception.
            # The exception is thrown because loosing a test is potentially a major issue and should be addressed ASAP
            # The threshold is somewhat arbitrary, however it is set to be large, we don't want to quit execution
            # while a valid test is executing.
            logger.info('Running Evaluation')
            episode_artifacts: typing.Dict[int, EpisodeArtifact] = {}
            # generator = ParallelRollouts(trainer.workers, mode='async')  # type: ignore
            # sample_data = ray.get([w.sample.remote() for w in trainer.workers.remote_workers()])
            # sample_data = concat_samples(sample_data)

            num_test_cases = self.test_case_manager.get_num_test_cases()
            test_status_scheduled = {i: False for i in range(0, num_test_cases)}
            num_episodes_left = len(test_status_scheduled)
            print(f"running evaluation rollouts, running {num_episodes_left} test cases")
            # Ideally we just call
            # sample_data = synchronous_parallel_sample(worker_set=trainer.workers)
            # however that call will just try to grab a full batch as opposed to just the
            # number of episodes we want, so we will call it's logic manually
            # this may not be required if batch_mode = complete episodes? this might also break
            # other algorithms such as SAC?
            worker_set = algorithm.workers
            all_sample_batches = []
            while num_episodes_left > 0:
                if algorithm.workers.num_remote_workers() <= 0:
                    sample_batches = [process_single_episode(worker_set.local_worker())]
                else:
                    remote_workers_to_use = list(range(1, min(algorithm.workers.num_remote_workers(), num_episodes_left) + 1))
                    sample_batches = worker_set.foreach_worker_with_id(
                        lambda _id,
                        w: process_single_episode(w),
                        local_worker=False,
                        healthy_only=True,
                        remote_worker_ids=remote_workers_to_use
                    )
                    if worker_set.num_healthy_remote_workers() <= 0:
                        # There is no point staying in this loop, since we will not be able to
                        # get any new samples if we don't have any healthy remote workers left.
                        break

                num_episodes_left -= len(sample_batches)
                print(f"{num_episodes_left} episodes left to run")
                all_sample_batches.extend(sample_batches)

            sample_data = concat_samples(all_sample_batches)

            print("finished running rollouts")

            test_status_scheduled = {i: False for i in range(0, num_test_cases)}

            temp_list = list(test_status_scheduled.values())
            # print(f"Processed {test_cases_processed} of {temp_list_len} test cases ---> ({percent_done}%)!!!")
            # sample_data = next(generator)

            # Iterate over the completed batches and update our scheduled/completed states accordingly
            for policy in sample_data.policy_batches.values():
                for episode_artifact in policy['eval']:
                    if episode_artifact.test_case is not None:
                        # Check to see if new test case for this item... If new process it results
                        if episode_artifact.test_case in test_status_scheduled and not test_status_scheduled[episode_artifact.test_case]:
                            # print the final episode state --- DEBUG --- TODO move to logger or remove
                            for key in episode_artifact.episode_state.keys():
                                logger.debug(f"{key}:{episode_artifact.test_case}:{episode_artifact.episode_state[key]}")
                            # Save off the artifacts and update scheduled state
                            episode_artifacts[episode_artifact.test_case] = episode_artifact
                            test_status_scheduled[episode_artifact.test_case] = True
                            temp_list = list(test_status_scheduled.values())
                            print(
                                f"Test case {episode_artifact.test_case} processed "
                                f"{np.count_nonzero(temp_list)} / {len(temp_list)} test cases."
                            )
                        # Check to see if we have already saved a test case for this item... If so warn
                        elif episode_artifact.test_case in test_status_scheduled and test_status_scheduled[episode_artifact.test_case]:
                            logger.debug("*" * 100)
                            t1 = {"environment." + k: v for k, v in episode_artifacts[episode_artifact.test_case].params.items()}
                            t2 = {"environment." + k: v for k, v in episode_artifact.params.items()}
                            logger.debug(
                                f"Warning - we have already processed this test case index --- Index = {episode_artifact.test_case}"
                            )
                            t = {k: (v, t2[k]) for k, v in t1.items()}
                            logger.debug(t)
                            logger.debug("*" * 100)

            # # Convert the raw output to dataframes of interest
            # logger.info('Formatting as a DataFrame')
            # episode_outputs, step_outputs = _convert_output_to_dataframe(raw_output, test_cases_num=list(test_cases.index))

            # get test_cases
            test_cases = self.test_case_manager.get_test_cases()

            # Generate our outcome object
            outcome = EvaluationOutcome(test_cases, episode_artifacts)

            # Do recording
            records: typing.List[IRecord] = []
            for reporter in self.recorders:
                record = reporter.resolve()
                record.save(outcome)
                records.append(record)

            return records
