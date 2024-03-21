"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator

from corl.parsers.yaml_loader import load_file


def try_load_file(value: dict[str, Any] | (str | Path)) -> dict[str, Any]:
    if isinstance(value, str | Path):
        return load_file(value)
    return value


class CorlPlatformConfigArgs(BaseModel):
    """
    name: name of the platform
    config: path to the config file this platform should be configured with
    """

    name: str
    config: Annotated[dict[str, Any], BeforeValidator(try_load_file)]


class CorlPlatformsConfigArgs(BaseModel):
    """
    platform_config: list of platforms
    """

    platform_config: list[CorlPlatformConfigArgs]


class CorlAgentConfigArgs(BaseModel):
    """
    name: name of the agent
    platforms: a list of platforms the agent at least partially controls
    config: path to the agent config for this agent
    policy: path to the policy config for this agent
    """

    name: str
    platforms: list[str]
    config: Annotated[dict[str, Any], BeforeValidator(try_load_file)]
    policy: Annotated[dict[str, Any], BeforeValidator(try_load_file)]
