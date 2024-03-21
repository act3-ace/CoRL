"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import difflib
import gc
import pprint
from functools import partial
from pathlib import Path
from typing import Annotated
from weakref import CallableProxyType, proxy

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from ray.rllib.algorithms.algorithm import Algorithm

from corl.evaluation.evaluation_factory import FactoryLoader
from corl.evaluation.recording.i_recorder import IRecorder
from corl.evaluation.runners.section_factories.engine.rllib.rllib_trainer import RllibConfig
from corl.evaluation.runners.section_factories.plugins.plugins import Plugins
from corl.evaluation.runners.section_factories.task import Experiment, Task
from corl.evaluation.runners.section_factories.teams import LoadableCorlAgent, Teams
from corl.evaluation.runners.section_factories.test_cases.test_case_manager import NoTestCases, TestCaseStrategy
from corl.libraries.algorithm_helper import cleanup_algorithm
from corl.libraries.factory import Factory


def factory(v, info, **kwargs):
    """Factory helper function to auto-generate objects"""
    obj_type = kwargs["obj_type"]
    if isinstance(v, obj_type):
        return v

    try:
        return Factory.resolve_factory({"type": obj_type, "config": v}, info=info)
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{e}") from e


class EvalExperiment(BaseModel):
    """Configures/validates inference setup"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    teams: Annotated[Teams, BeforeValidator(partial(factory, obj_type=Teams))]
    task: Annotated[Task, BeforeValidator(partial(factory, obj_type=Task))]
    plugins: Annotated[Plugins, BeforeValidator(partial(factory, obj_type=Plugins))] | None = None
    engine: Annotated[RllibConfig, BeforeValidator(partial(factory, obj_type=RllibConfig))]
    test_case_manager: Annotated[TestCaseStrategy, BeforeValidator(Factory.resolve_factory)] = Field(default_factory=NoTestCases)

    recorders: list[Annotated[IRecorder, BeforeValidator(FactoryLoader.resolve_factory)]] = Field(default=[], validate_default=True)

    cfg: Path | None = None
    output_dir: Path | None = None

    experiment_: Experiment | None = None
    algorithm_: Algorithm | None = None

    @property
    def experiment(self) -> Experiment:
        if not self.experiment_:
            self.experiment_ = Experiment(teams=self.teams, task=self.task)  # type: ignore
        return self.experiment_

    @property
    def algorithm(self) -> CallableProxyType:
        if not self.algorithm_:
            raise RuntimeError("Algorithm not created")
        return proxy(self.algorithm_)

    def __enter__(self):
        if self.algorithm_ is None:
            """Build an algorithm from engine/experiment/plugins"""

            try:
                algorithm = self.engine.rllib.generate(
                    self.experiment,
                    test_case_manager=self.test_case_manager,
                    plugins=self.plugins,
                )
            except ValueError:
                raise
            except Exception as e:  # noqa: BLE001
                raise ValueError from e

            # Now insert each participant's weights into the algorithm
            def apply_weights(agent: LoadableCorlAgent):
                try:
                    agent.agent_loader.apply_to_algorithm(algorithm, agent.name)
                except Exception as e:  # noqa: BLE001
                    err = f"{e}"
                    if agent.agent_loader.env_config:
                        trained_glues = agent.agent_loader.env_config["agents"][agent.agent_loader.agent_id].class_config.config["glues"]
                        inference_glues = algorithm.config["env_config"]["agents"][agent.name].class_config.config["glues"]

                        trained_glue_config = pprint.pformat(trained_glues).split("\n")
                        inference_glue_config = pprint.pformat(inference_glues).split("\n")

                        html_diff = difflib.HtmlDiff().make_file(trained_glue_config, inference_glue_config)
                        text_diff = "\n".join(difflib.ndiff(trained_glue_config, inference_glue_config))

                        diff_path = Path(f"{agent.name}_diff.html")
                        diff_path.write_text(html_diff, encoding="UTF-8")
                        err += f"\nAgent config changes between training and inference for {agent.name} saved to '{diff_path}'\n{text_diff}"

                    raise ValueError(f"{err}") from e

            self.teams.iterate_on_participant(apply_weights)
            if algorithm.workers is not None:
                algorithm.workers.sync_weights()

            self.algorithm_ = algorithm

        return self.algorithm

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.algorithm_ is not None:
            cleanup_algorithm(self.algorithm_)
            self.algorithm_ = None
            gc.collect()
