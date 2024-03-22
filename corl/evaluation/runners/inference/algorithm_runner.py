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
import traceback
import typing
from collections.abc import Iterator
from threading import Condition
from weakref import CallableProxyType

from pydantic import BaseModel
from ray.rllib.env.base_env import BaseEnv, convert_to_base_env
from ray.rllib.evaluation.env_runner_v2 import EnvRunnerV2, _build_multi_agent_batch, _PerfStats
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils.typing import EnvID, SampleBatchType

from corl.dones.done_func_base import DoneStatusCodes
from corl.environment.multi_agent_env import ACT3MultiAgentEnv
from corl.evaluation.runners.section_factories.engine.rllib.interruptable_callback import InterruptableCallback


class AlgorithmRunnerValidator(BaseModel):
    """Validator for AlgorithmRunner"""

    exit_on_end: bool = False


class AlgorithmRunner:
    """Runs an environment"""

    def __init__(self, **kwargs):
        """"""
        self.config = self.get_validator()(**kwargs)

        self._algorithm: CallableProxyType | None = None  # type: ignore
        self._base_env: BaseEnv | None = None  # type: ignore
        self._running: bool = True  # type: ignore
        self.__runner: EnvRunnerV2 | None = None  # type: ignore

        self._logger = logging.getLogger(AlgorithmRunner.__name__)

        self._algorithm_cv = Condition()

    @staticmethod
    def get_validator() -> type[AlgorithmRunnerValidator]:
        """Return the validator for AlgorithmRunner"""
        return AlgorithmRunnerValidator

    @property
    def algorithm(self) -> CallableProxyType:
        if self._algorithm is None:
            raise RuntimeError("algorithm has not been initialized")
        return self._algorithm

    @property
    def base_env(self) -> BaseEnv:
        return convert_to_base_env(env=self.local_worker.env)

    @property
    def local_worker(self) -> RolloutWorker:
        if self.algorithm.workers is None:
            raise RuntimeError("No Algorithm workers")

        return self.algorithm.workers.local_worker()

    def reset(self, algorithm: CallableProxyType | None = None):
        """Stops any existing episodes and sets the algorithm used by this _runner"""

        with self._algorithm_cv:
            self._logger.info(f'{"*" * 40}    RESETTING    {"*" * 40}')
            if algorithm is not None:
                algorithm = typing.cast(CallableProxyType, algorithm)  # type: ignore

                self.__runner = None

                # if self._algorithm is not None:
                #     # HACK: Cleanup simulator simulator
                #     def shutdown_env(env):
                #         try:
                #             env.simulator.shutdown()
                #         except (RuntimeError, KeyError, AttributeError):
                #             pass

                #     self._algorithm.workers.foreach_env(shutdown_env)
                #     self._algorithm.stop()
                #     del self._algorithm
                self._algorithm = algorithm

            gc.collect()
            self._algorithm_cv.notify_all()

    @property
    def _runner(self) -> EnvRunnerV2:
        if self.__runner is None:
            multiple_episodes_in_batch = self.algorithm.config.batch_mode != "complete_episodes"
            rollout_fragment_length = int(self.algorithm.config.rollout_fragment_length)
            self.__runner = EnvRunnerV2(
                self.local_worker,
                convert_to_base_env(env=self.local_worker.env),
                multiple_episodes_in_batch=multiple_episodes_in_batch,
                callbacks=self.algorithm.callbacks,
                perf_stats=_PerfStats(),
                rollout_fragment_length=rollout_fragment_length,
            )
        return self.__runner

    def _handle_errored_episode(self, env_id: EnvID, error: Exception) -> Iterator[SampleBatchType]:
        outputs: list[SampleBatch | MultiAgentBatch] = []
        episode: EpisodeV2 = self._runner._active_episodes[env_id]  # noqa: SLF001
        batch_builder = self._runner._batch_builders[env_id]  # noqa: SLF001

        env = self.local_worker.env
        if isinstance(env, ACT3MultiAgentEnv):
            metadata = {}
            metadata.update(env.config.metadata)
            metadata["error"] = str(traceback.format_exc())

            for platform_name in env.state.episode_state:
                env._state.episode_state[platform_name][f"{type(error).__name__}"] = DoneStatusCodes.DRAW  # noqa: SLF001

            env.simulator.mark_episode_done(env._done_info, env._state.episode_state, metadata)  # noqa: SLF001

        episode.postprocess_episode(
            batch_builder=batch_builder,
            is_done=True,
            check_dones=False,
        )

        ma_sample_batch = _build_multi_agent_batch(
            episode.episode_id,
            batch_builder,
            self._runner._large_batch_threshold,  # noqa: SLF001
            self._runner._multiple_episodes_in_batch,  # noqa: SLF001
        )
        if ma_sample_batch:
            outputs.append(ma_sample_batch)

        # SampleBatch built from data collected by batch_builder.
        # Clean up and delete the batch_builder.
        del self._runner._batch_builders[env_id]  # noqa: SLF001

        yield from outputs

    def run(self) -> Iterator[SampleBatchType]:
        def process_outputs(outputs):
            if isinstance(
                outputs,
                SampleBatch | MultiAgentBatch,
            ):
                output = concat_samples([outputs])

                self.local_worker.callbacks.on_sample_end(worker=self.local_worker, samples=output)

                return output
            return None

        while self._running:
            with self._algorithm_cv:
                try:
                    self._logger.debug("Running...")
                    for outputs in self._runner.run():
                        if output := process_outputs(outputs):
                            yield output
                            break

                except Exception as exception:  # noqa: BLE001
                    self._logger.error(exception)
                    for outputs in self._handle_errored_episode(env_id=0, error=exception):
                        if output := process_outputs(outputs):
                            yield output
                finally:
                    self.__runner = None

                    if self.config.exit_on_end:
                        self._running = False

    def stop(self, continue_running=False):
        try:
            for callback in self.algorithm.callbacks._callback_list:  # noqa: SLF001
                if isinstance(callback, InterruptableCallback):
                    callback.stop()

            if not continue_running:
                self._running = False
                self.local_worker.stop()
        except (ReferenceError, RuntimeError):
            pass
