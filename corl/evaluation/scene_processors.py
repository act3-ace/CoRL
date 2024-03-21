"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import dataclasses

from corl.evaluation import metrics
from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.metrics.processors import Evaluation
from corl.evaluation.metrics.scenario_alert_generators import ScenarioAlertGenerators
from corl.evaluation.metrics.scenario_metric_generators import ScenarioMetricGenerators


@dataclasses.dataclass
class SceneProcessors:
    """Contains the processors for a scene.

    Here "Scene" refers to processors in the world and processors attached to participants
    """

    world: Evaluation
    participants: dict[str, Evaluation]

    @staticmethod
    def from_event(
        event_data: EpisodeArtifact,
        scenario_metric_generators: ScenarioMetricGenerators,
        scenario_alert_generators: ScenarioAlertGenerators,
    ) -> "SceneProcessors":
        """Post Process an Event

        This method is not implemented
        """
        raise NotImplementedError()

    @staticmethod
    def from_evaluation(
        evaluation_data: list[EpisodeArtifact],
        scenario_metric_generators: ScenarioMetricGenerators,
        scenario_alert_generators: ScenarioAlertGenerators,
    ) -> "SceneProcessors":
        """Post process and evaluation

        Args:
            evaluation_data (typing.List[EpisodeArtifact]): Data from evaluation
            scenario_metric_generators (ScenarioMetricGenerators): metric generators
            scenario_alert_generators (ScenarioAlertGenerators): alert generators

        Returns:
            _type_: _description_
        """

        contained_participants: list[str] = []
        for episode_artifact in evaluation_data:
            for agent_id in episode_artifact.steps[0].agents:
                if agent_id not in contained_participants:
                    contained_participants.append(agent_id)

        # Process any world metrics
        world_evaluation_metrics: Evaluation = metrics.processors.Evaluation.process(
            evaluation_data,
            scenario_metric_generators.world,
            scenario_alert_generators.world,
        )

        # Processes each agent's metrics
        participant_evaluation_metrics: dict[str, Evaluation] = {}
        for agent_id in contained_participants:
            # Find the generator to use
            if agent_id in scenario_metric_generators.specific_participant:
                generator = scenario_metric_generators.specific_participant[agent_id]
            elif scenario_metric_generators.default_participant is not None:
                generator = scenario_metric_generators.default_participant
            else:
                raise RuntimeError(f"A generator could not be established to generate metrics for {agent_id}")

            # Find the generator to use
            if agent_id in scenario_metric_generators.specific_participant:
                if agent_id not in scenario_alert_generators.specific_participant:
                    scenario_alert_generators.specific_participant[agent_id] = []

                alert_generator = scenario_alert_generators.specific_participant[agent_id]
            elif scenario_metric_generators.default_participant is not None:
                alert_generator = scenario_alert_generators.default_participant
            else:
                raise RuntimeError(f"A generator could not be established to generate metrics for {agent_id}")

            # process the metrics
            participant_evaluation_metrics[agent_id] = metrics.processors.Evaluation.process(
                evaluation_data,
                generator,
                alert_generator,
                agent_id=agent_id,
            )

        return SceneProcessors(world_evaluation_metrics, participant_evaluation_metrics)
