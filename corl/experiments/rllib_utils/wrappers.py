"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from ray.tune.registry import ENV_CREATOR, _global_registry


def get_rllib_environment_creator(env_specifier):
    """
    Returns CoRL environment
    """
    if _global_registry.contains(ENV_CREATOR, env_specifier):
        return _global_registry.get(ENV_CREATOR, env_specifier)

    raise RuntimeError(f"No registered env creator named {env_specifier}")
