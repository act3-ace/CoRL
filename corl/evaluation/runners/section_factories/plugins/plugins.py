"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from corl.evaluation.default_config_updates import DoNothingConfigUpdate
from corl.evaluation.evaluation_factory import FactoryLoader

from .config_updater import ConfigUpdate
from .environment_state_extractor import EnvironmentStateExtractor
from .platform_serializer import PlatformSerializer


class Plugins(BaseModel):
    """Plugins that define environment specific information to run an evaluation"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    platform_serialization: Annotated[PlatformSerializer, BeforeValidator(FactoryLoader.resolve_factory)]
    eval_config_update: list[Annotated[ConfigUpdate, BeforeValidator(FactoryLoader.resolve_factory)]] = Field(
        validate_default=True, default_factory=lambda: [DoNothingConfigUpdate()]
    )

    environment_state_extractor: Annotated[EnvironmentStateExtractor, BeforeValidator(FactoryLoader.resolve_factory)] | None = None
