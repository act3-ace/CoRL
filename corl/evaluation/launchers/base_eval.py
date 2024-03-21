from abc import abstractmethod
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch

from corl.evaluation.launchers.evaluate_validator import EvalExperiment
from corl.libraries.factory import Factory
from corl.parsers.yaml_loader import load_file


def get_default_evaluator() -> "BaseEvaluator":
    """Get the default evaluator type"""
    from corl.evaluation.runners.iterate_test_cases import TestCaseEvaluator

    return TestCaseEvaluator()


def get_default_processors() -> list["BaseProcessor"]:
    """Get the default processor type"""
    from corl.evaluation.runners.iterate_test_cases import ConfigSaver, SampleBatchProcessor

    return [SampleBatchProcessor(), ConfigSaver()]


def load_raw_config(v, info, **kwargs):
    if v is None or (isinstance(v, dict) and len(v) == 0):
        if info.data["path"] is None:
            raise ValueError("'path' or 'config' must be set.")
        return load_file(config_filename=info.data["path"])
    return v


def from_raw_config(key, v, info, **kwargs):
    config = info.data["raw_config"].get(key, v)

    return Factory.resolve_factory(config, info, **kwargs)


def processors_from_raw_config(v, info, **kwargs):
    processors = info.data["raw_config"].get("processors", v)

    return [Factory.resolve_factory(processor, info, **kwargs) for processor in processors]


class EvalConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path | None = None
    raw_config: Annotated[dict | None, BeforeValidator(load_raw_config)] = None

    evaluator: Annotated["BaseEvaluator", BeforeValidator(partial(from_raw_config, "evaluator"))] = Field(
        default_factory=get_default_evaluator, validate_default=True
    )
    processors: Annotated[list["BaseProcessor"], BeforeValidator(processors_from_raw_config)] = Field(
        default_factory=get_default_processors, validate_default=True
    )

    experiment: Annotated[EvalExperiment, BeforeValidator(partial(from_raw_config, "experiment"))] = Field(  # type: ignore
        default=None, validate_default=True
    )


class BaseEvaluatorValidator(BaseModel):
    """Base class validator for BaseEvaluator"""

    ...


class BaseEvaluator:
    """Base class for evaluator objects"""

    def __init__(self, **kwargs):
        self._config = self.get_validator()(**kwargs)

    @staticmethod
    def get_validator() -> type[BaseEvaluatorValidator]:
        """The validator for this class"""
        return BaseEvaluatorValidator

    @abstractmethod
    def __call__(self, config: EvalConfig) -> Iterator[SampleBatch | MultiAgentBatch]:
        """Generates results, which may be passed to a Processor"""
        ...

    @abstractmethod
    def reset(self):
        """Resets evaluation"""
        ...

    @abstractmethod
    def stop(self):
        """Stops evaluation"""
        ...


class BaseProcessorValidator(BaseModel):
    """Base class validator for BaseProcessor"""

    ...


class BaseProcessor:
    """Base class for result processor objects"""

    def __init__(self, **kwargs):
        self._config = self.get_validator()(**kwargs)

    @staticmethod
    def get_validator() -> type[BaseProcessorValidator]:
        """The validator for this class"""
        return BaseProcessorValidator

    @abstractmethod
    def __call__(self, config: EvalConfig, results: Any) -> Any:
        """Processes results from an Evaluator"""
        ...

    @abstractmethod
    def stop(self):
        """Stops the processor"""
        ...
