# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

from functools import cached_property

import gymnasium
import numpy as np
import pytest
from pydantic import ValidationError

from corl.glues.base_glue import BaseAgentGlue, TrainingExportBehavior


class TestGlue(BaseAgentGlue):
    @cached_property
    def action_space(self) -> gymnasium.spaces.Space:
        raise Exception("Unimplemented")

    def apply_action(
        self,  # noqa: PLR6301
        action: np.ndarray | tuple | dict,
        observation: np.ndarray | tuple | dict,
        action_space,
        observation_space,
        observation_units,
    ) -> None:
        raise Exception("Unimplemented")

    @cached_property
    def observation_space(self) -> gymnasium.spaces.Space:
        raise Exception("Unimplemented")

    def get_observation(self, other_obs, obs_space, obs_units) -> np.ndarray | tuple | dict:  # noqa: PLR6301
        raise Exception("Unimplemented")

    def get_unique_name(self) -> str:  # noqa: PLR6301
        raise Exception("Unimplemented")


def test_training_export_behavior_parse():
    config = {"agent_name": "blue0", "training_export_behavior": "INCLUDE"}

    test_glue = TestGlue(**config)
    assert test_glue.config.training_export_behavior == TrainingExportBehavior.INCLUDE

    config = {"agent_name": "blue0", "training_export_behavior": "EXCLUDE"}

    test_glue = TestGlue(**config)
    assert test_glue.config.training_export_behavior == TrainingExportBehavior.EXCLUDE

    config = {"agent_name": "blue0", "training_export_behavior": "INVALID"}

    with pytest.raises(ValidationError):
        test_glue = TestGlue(**config)
