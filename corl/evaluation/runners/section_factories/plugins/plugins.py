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
import typing

import jsonargparse
from pydantic.dataclasses import dataclass

from .config_updater import ConfigUpdate  # type: ignore
from .environment_state_extractor import EnvironmentStateExtractor
from .platform_serializer import PlatformSerializer


class Config:
    """Pydantic options"""
    arbitrary_types_allowed = True


@jsonargparse.typing.final
@dataclass(config=Config)
class Plugins:  # pylint: disable=R0903
    """Plugins that define environment specific information to run an evaluation
    """
    platform_serialization: PlatformSerializer
    eval_config_update: typing.List[ConfigUpdate] = dataclasses.field(
        default_factory=lambda: ['corl.evaluation.default_config_updates.DoNothingConfigUpdate']  # type: ignore
    )
    environment_state_extractor: typing.Optional[EnvironmentStateExtractor] = None
