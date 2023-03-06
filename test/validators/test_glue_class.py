"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import gym
import numpy as np
from pydantic import ValidationError
import pytest
import typing

from corl.glues.base_glue import BaseAgentGlue, TrainingExportBehavior


class TestGlue(BaseAgentGlue):
    def action_space(self) -> gym.spaces.Space:
        raise Exception("Unimplemented")

    def apply_action(self, action: typing.Union[np.ndarray, typing.Tuple, typing.Dict], observation: typing.Union[np.ndarray, typing.Tuple, typing.Dict], action_space, observation_space, observation_units) -> None:
        raise Exception("Unimplemented")

    def observation_space(self) -> gym.spaces.Space:
        raise Exception("Unimplemented")

    def get_observation(self, other_obs, obs_space, obs_units) -> typing.Union[np.ndarray, typing.Tuple, typing.Dict]:
        raise Exception("Unimplemented")

    def get_unique_name(self) -> str:
        raise Exception("Unimplemented")


def test_training_export_behavior_parse():

    config = {
        "agent_name": "blue0",
        "training_export_behavior": "INCLUDE"
    }

    test_glue = TestGlue(**config)
    assert(test_glue.config.training_export_behavior == TrainingExportBehavior.INCLUDE)

    config = {
        "agent_name": "blue0",
        "training_export_behavior": "EXCLUDE"
    }

    test_glue = TestGlue(**config)
    assert(test_glue.config.training_export_behavior == TrainingExportBehavior.EXCLUDE)

    config = {
        "agent_name": "blue0",
        "training_export_behavior": "INVALID"
    }

    with pytest.raises(ValidationError):
        test_glue = TestGlue(**config)
