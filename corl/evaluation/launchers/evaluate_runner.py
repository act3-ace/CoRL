"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import gc
import logging
import os
import tempfile
import time
from pathlib import Path
from threading import Condition, Thread

from pydantic import BaseModel, ConfigDict
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch

from corl.evaluation.connection.base_eval_connection import BaseEvalConnection, EvalConfig
from corl.evaluation.connection.signal import Slot


class EvalThreadValidator(BaseModel):
    eval_config: EvalConfig


class EvalThread(Thread):
    def __init__(self, config: dict, **kwargs):
        super().__init__(**kwargs)
        self._config = EvalThreadValidator(**config)
        self._logger = logging.getLogger(type(self).__name__)

    def run(self):
        def process_batch(batch: SampleBatch | MultiAgentBatch):
            to_process = [batch]
            for needs_processing in to_process:
                for processor in self._config.eval_config.processors:
                    try:
                        if new_result := processor(self._config.eval_config, needs_processing):
                            to_process.append(new_result)
                    except Exception as e:  # noqa: PERF203
                        self._logger.exception(e)

        # from concurrent.futures import ThreadPoolExecutor
        # with self._config.eval_schema.evaluator_config:
        #     with ThreadPoolExecutor(2) as executor:
        #         for result in self._config.evaluator(self._config.eval_schema):
        #             executor.submit(process_batch, result)

        with self._config.eval_config.experiment:
            for result in self._config.eval_config.evaluator(self._config.eval_config):
                self._logger.debug("Processing result")
                process_batch(result)

    def stop(self):
        self._config.eval_config.evaluator.stop()

    def reset(self):
        self._config.eval_config.evaluator.reset()


class EvalRunnerValidator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_forever: bool = False
    tmpdir_base: Path | None = None
    connection: BaseEvalConnection | None = None


class EppReset:
    ...


class EvalRunner:
    def __init__(self, **kwargs):
        super().__init__()
        self._config = self.get_validator()(**kwargs)

        if self._config.connection is not None:
            self._config.connection.reset_signal.register(Slot(self))
            self._config.connection.modify_epp_signal.register(Slot(lambda x: self(EppReset())))

        self._logger = logging.getLogger(type(self).__name__)

        self._thread: EvalThread | None = None  # type: ignore
        self._running = True
        self._condition = Condition()
        self._next_eval_config: EvalConfig | None = None  # type: ignore

    @staticmethod
    def get_validator() -> type[EvalRunnerValidator]:
        return EvalRunnerValidator

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.stop()

    def _on_eval_config(self, message: EvalConfig):
        self._logger.debug("Setting next_schema")
        with self._condition:
            self._next_eval_config = message
            if self._thread:
                self._logger.debug("Stopping thread")
                self._thread.stop()

    def _on_epp_reset(self):
        self._logger.debug("Restarting thread...")
        with self._condition:
            if self._thread:
                self._thread.reset()

    def __call__(self, message: EvalConfig | EppReset):
        if isinstance(message, EvalConfig):
            return self._on_eval_config(message)

        if isinstance(message, EppReset):
            return self._on_epp_reset()

        raise RuntimeError(f"Unhandled message type {type(message)}")

    def _run_loop(self, output_dir: Path | None = None):
        count = 0
        while self._running:
            if count >= 1 and not self._config.run_forever:
                break

            if self._next_eval_config:
                with self._condition:
                    self._logger.debug("Get lock")
                    if self._next_eval_config:
                        count += 1

                        if output_dir:
                            self._next_eval_config.experiment.engine.rllib.output_dir = output_dir

                        self._logger.debug("Creating new EvalThread")
                        self._thread = EvalThread(config={"eval_config": self._next_eval_config}, name=f"{type(self).__name__}{count}")
                        self._next_eval_config = None
                        self._thread.start()
                if self._thread:
                    self._logger.debug("Thread join...")
                    self._thread.join()
                    del self._thread
                    gc.collect()
                    self._logger.debug("Thread joined")

            else:
                self._logger.debug("Waiting for next_schema...")
                time.sleep(1)

    def run(self):
        if self._config.tmpdir_base:
            os.makedirs(self._config.tmpdir_base, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self._config.tmpdir_base) as output_dir:
                self._run_loop(Path(output_dir))
        else:
            self._run_loop()
