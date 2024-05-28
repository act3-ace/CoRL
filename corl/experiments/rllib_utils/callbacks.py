# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
"""
Utilities to support RlLib callbacks
"""

import logging

from ray.actor import ActorHandle


def get_corl_sub_env(base_env, episode):
    """
    Get corl sub env. RLLibs envs can be many different types and have a consistent api.
    This cause problems we when attempt to make a consistent API for callbacks.

    Currently the callbacks do not support ActorHandles. In the future we will need to
    adjust callbacks based on env type.

    Per RlLib:
        EnvType represents a BaseEnv, MultiAgentEnv, ExternalEnv, ExternalMultiAgentEnv,
        VectorEnv, gym.Env, or ActorHandle.
        (sven): Specify this type more strictly (it should just be gym.Env).

    """
    env = base_env.get_sub_environments()[episode.env_id]

    if isinstance(env, ActorHandle):
        logging.warn("ActorHandle is not supported in callbacks")
        return None

    return env
