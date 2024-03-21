# mypy: disable-error-code="union-attr"
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
import logging
import os
import shutil
from collections import defaultdict
from collections.abc import Iterator
from distutils.dir_util import copy_tree
from pathlib import Path
from threading import Condition
from typing import Any
from weakref import CallableProxyType

import ray.cloudpickle as pickle
import yaml
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.util.debug import disable_log_once_globally

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.evaluation_outcome import EvaluationOutcome
from corl.evaluation.launchers.base_eval import BaseEvaluator, BaseProcessor, EvalConfig
from corl.evaluation.launchers.evaluate_validator import EvalExperiment
from corl.evaluation.loader.policy_checkpoint import PolicyCheckpoint
from corl.evaluation.recording.i_recorder import IRecord
from corl.evaluation.runners.section_factories.engine.rllib.rllib_trainer import ray_context
from corl.evaluation.runners.section_factories.test_cases.default_strategy import DefaultStrategy
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import TestCaseStrategy


def process_single_episode(worker: RolloutWorker):
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


class TestCaseEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = logging.getLogger(type(self).__name__)
        self._condition = Condition()
        self._running = True

        self._algorithm: CallableProxyType | None = None  # type: ignore

    def __call__(self, config: EvalConfig) -> Iterator[SampleBatch | MultiAgentBatch]:
        evaluator_config = config.experiment
        """Iterates and evaluates all tests cases"""
        # Disable logging because can cause slowdowns
        disable_log_once_globally()

        # The following context must be valid during the running of parallel workers and saving of data
        # Important that this stay here.
        with ray_context(local_mode=evaluator_config.engine.rllib.debug_mode, include_dashboard=False):
            # Do evaluation using ParallelRollouts
            # See Issue https://git.aoc-pathfinder.cloud//ML/act3-rllib-agents/-/issues/1423 for more discussion on this block
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
            # The exception is thrown because losing a test is potentially a major issue and should be addressed ASAP
            # The threshold is somewhat arbitrary, however it is set to be large, we don't want to quit execution
            # while a valid test is executing.
            self.logger.info("Running Evaluation")

            with self._condition:
                if not self._running:
                    self._algorithm = None
                    return
                self._algorithm = evaluator_config.algorithm

            num_test_cases = evaluator_config.test_case_manager.get_num_test_cases()

            test_status_scheduled: dict[int, bool] = {i: False for i in range(num_test_cases)}
            num_episodes_left = len(test_status_scheduled)
            self.logger.info(f"Running evaluation rollouts, running {num_episodes_left} test cases")
            # Ideally we just call
            # sample_data = synchronous_parallel_sample(worker_set=trainer.workers)
            # however that call will just try to grab a full batch as opposed to just the
            # number of episodes we want, so we will call it's logic manually
            # this may not be required if batch_mode = complete episodes? this might also break
            # other algorithms such as SAC?
            worker_set = self._algorithm.workers
            if worker_set is None:
                raise RuntimeError("No workers available")

            sample_batches: list[SampleBatch | MultiAgentBatch]
            while num_episodes_left > 0:
                if worker_set.num_remote_workers() <= 0:
                    sample_batches = [process_single_episode(worker_set.local_worker())]
                else:
                    remote_workers_to_use = worker_set.healthy_worker_ids()
                    sample_batches = worker_set.foreach_worker_with_id(
                        lambda _id, w: process_single_episode(w),
                        local_worker=False,
                        healthy_only=True,
                        remote_worker_ids=remote_workers_to_use,
                    )
                    if worker_set.num_healthy_remote_workers() <= 0:
                        # There is no point staying in this loop, since we will not be able to
                        # get any new samples if we don't have any healthy remote workers left.
                        break

                with self._condition:
                    if not self._running:
                        return

                    num_episodes_left -= min(len(sample_batches), num_episodes_left)

                    self.logger.info(f"{num_episodes_left} episodes left to run")

                yield concat_samples(sample_batches)

            self.logger.info("Finished running rollouts")
            with self._condition:
                self._algorithm = None

    def reset(self):
        self.logger.warn("Reset not supported. Continuing without resetting...")

    def stop(self):
        with self._condition:
            self._running = False
            worker_set = self._algorithm.workers if self._algorithm is not None else None
            if worker_set is not None:
                worker_set.stop()


class SampleBatchProcessor(BaseProcessor):
    """Converts SampleBatches into records"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger = logging.getLogger(type(self).__name__)
        self._running = True

    def __call__(self, config: EvalConfig, results: Any) -> list[IRecord]:
        if not self._running:
            return []

        if not isinstance(
            results,
            SampleBatch | MultiAgentBatch,
        ):
            return []

        return self.process_sample_data(config.experiment, results)

    def stop(self):
        self._running = False

    def process_sample_data(self, config: EvalExperiment, sample_data: SampleBatch | MultiAgentBatch) -> list[IRecord]:
        episode_artifacts: defaultdict[int, list[EpisodeArtifact]] = defaultdict(list)

        if isinstance(sample_data, SampleBatch):
            self.process_sample_batch(config.test_case_manager, sample_data, episode_artifacts)

        elif isinstance(sample_data, MultiAgentBatch):
            for sample_batch in sample_data.policy_batches.values():
                self.process_sample_batch(config.test_case_manager, sample_batch, episode_artifacts)

        # Generate our outcome object
        if isinstance(config.test_case_manager, DefaultStrategy):
            # DefaultStrategy does not plan test cases, so the test case initial conditions need to be manually
            # added to the test case list after the episode to save the outcome
            for episode_artifact in episode_artifacts.values():
                for artifact in episode_artifact:
                    init_cond = {f"environment.{state}": value for state, value in artifact.params.items()}
                    init_cond["test_case"] = artifact.test_case
                    config.test_case_manager.test_cases.append(init_cond)

        outcome = EvaluationOutcome(config.test_case_manager.get_test_cases(), episode_artifacts)

        records: list[IRecord] = []
        for recorder in config.recorders:
            record = recorder.resolve()
            record.save(outcome)
            records.append(record)

        return records

    def process_sample_batch(
        self, test_case_manager: TestCaseStrategy, sample_batch: SampleBatch, episode_artifacts: defaultdict[int, list[EpisodeArtifact]]
    ):
        """Iterate over the completed batches and update our scheduled/completed states accordingly"""
        episode_artifact: EpisodeArtifact
        for episode_artifact in sample_batch["eval"]:
            if episode_artifact.test_case is not None:
                # Check to see if new test case for this item... If new process it results
                test_case_id = test_case_manager.get_test_case_index(episode_artifact.test_case)
                for key in episode_artifact.episode_state:
                    self.logger.debug(f"{key}:{episode_artifact.test_case}:{episode_artifact.episode_state[key]}")
                # Save off the artifacts and update scheduled state
                episode_artifacts[test_case_id].append(episode_artifact)


class ConfigSaver(BaseProcessor):
    """Saves env_config and checkpoints with records"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._evaluation_output_paths: list[Path] = []  # type: ignore
        self._running = True

    def stop(self):
        self._running = False

    def __call__(self, config: EvalConfig, results: Any):
        if not self._running:
            return

        if not isinstance(results, list):
            results = [results]
        if not all(isinstance(r, IRecord) for r in results):
            return

        result: IRecord
        for result in results:
            if result.absolute_path not in self._evaluation_output_paths:
                self.save_config(config, result.absolute_path)
                self._evaluation_output_paths.append(result.absolute_path)

    @staticmethod
    def save_config(config: EvalConfig, directory: Path):
        """Save env_config and checkpoints to output_path"""
        os.makedirs(directory, exist_ok=True)

        # Copies the configuration file
        config_filename = directory / config.path.name if config.path is not None else directory / "config.yml"

        with open(config_filename, "w") as file_obj:
            yaml.dump(config.raw_config, file_obj)

        env_config = config.experiment.experiment.env_config
        env_config_filename = directory / "env_config.pkl"
        with open(env_config_filename, "wb") as file_obj:
            pickle.dump(env_config, file_obj)

        # Copies the checkpoints for each agent
        checkpoint_dir = directory / "agent_checkpoints"
        os.makedirs(checkpoint_dir)

        for agent_config in config.experiment.teams.agent_config:
            if isinstance(agent_config.agent_loader, PolicyCheckpoint):
                checkpoint_filename = agent_config.agent_loader.checkpoint_filename
                if checkpoint_filename is not None:
                    agent_checkpoint_dir = checkpoint_dir / agent_config.name
                    os.makedirs(agent_checkpoint_dir)
                    if os.path.isdir(checkpoint_filename):
                        copy_tree(str(checkpoint_filename), str(agent_checkpoint_dir))
                    else:
                        shutil.copy(checkpoint_filename, agent_checkpoint_dir)
