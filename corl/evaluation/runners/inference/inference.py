"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from pydantic import validator

from corl.evaluation.launchers.base_eval import BaseEvaluator, BaseEvaluatorValidator, EvalConfig

from .algorithm_runner import AlgorithmRunner


class InferenceValidator(BaseEvaluatorValidator):
    class Config:
        arbitrary_types_allowed = True

    algorithm_runner: AlgorithmRunner = None  # type: ignore

    @validator("algorithm_runner", pre=True, always=True)
    def construct_algorithm_runner(cls, v):
        if isinstance(v, AlgorithmRunner):
            return v

        if isinstance(v, dict):
            return AlgorithmRunner(**v)

        return AlgorithmRunner()


class Inference(BaseEvaluator):
    def __init__(self, **kwargs):
        self._config: InferenceValidator  # type: ignore
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[InferenceValidator]:
        return InferenceValidator

    def reset(self):
        self._config.algorithm_runner.stop(continue_running=True)

    def stop(self):
        self._config.algorithm_runner.stop()

    def __call__(self, config: EvalConfig, **kwargs):
        self._config.algorithm_runner.reset(config.experiment.algorithm)
        yield from self._config.algorithm_runner.run()
