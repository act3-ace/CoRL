"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Library to extract the value function from within callbacks
"""

from collections import defaultdict

from ray.rllib import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


def extract_value_function_step_callback(
    *,
    worker,
    base_env: BaseEnv,
    policies: dict[PolicyID, Policy] | None = None,
    episode: EpisodeV2,
    **kwargs,
) -> dict[str, float]:
    """Extract the value function from information provided by on_episode_step"""

    if policies is None:
        return {}

    env = base_env.get_sub_environments()[episode.env_id]

    non_trainable_agent_dict = {k: v for (k, v) in env.agent_dict.items() if not (v.trainable)}
    # Map of policy_id to agent_id(s)
    policy_id_map = defaultdict(list)
    for agent_id in env.agent_to_platforms:
        if agent_id in non_trainable_agent_dict:
            continue
        t_policy_id = episode.policy_for(agent_id)
        policy_id_map[t_policy_id].append(agent_id)

    # Pull out state value for each agent from policy.model.value_function
    output_dict = {}
    for agent_id in env.agent_to_platforms:
        if agent_id in non_trainable_agent_dict:
            continue
        policy_id = episode.policy_for(agent_id)
        policy = policies[policy_id]

        # Tensor size is the num obs fed into the vf in a forward pass
        try:
            vf_tensor = policy.model.value_function().tolist()  # type: ignore
        except AttributeError:
            continue

        # Assume first value in tensor
        vf_value = vf_tensor[0]

        # Determine tensor values for agents sharing a policy
        if len(policy_id_map[policy_id]) > 1:
            # Pull out the set of observations used for the previous VF model forward pass
            # https://docs.ray.io/en/latest/rllib/package_ref/env/base_env.html#ray.rllib.env.base_env.BaseEnv.poll
            # (1) New observations for each ready agent
            # (2) Reward values for each ready agent. If the episode is just started, the value will be None.
            # (3) Terminated values for each ready agent. The special key “__all__” is used to indicate episode termination.
            # (4) Truncated values for each ready agent. The special key “__all__” is used to indicate episode truncation.
            # (5) Info values for each ready agent. Agents may take off-policy actions, in which case, there will be an
            #     entry in this dict that contains the taken action.
            # (6) There is no need to send_actions() for agents that have already chosen off-policy actions.
            trainable_observations, _, agent_termination_values, _, _, _ = base_env.poll()

            # Indicates episode ended, no way to identify owner of vf_value
            # Indicates agent's obs not provided in previous model forward pass - no value
            if len(trainable_observations) == 0 or agent_id not in trainable_observations[episode.env_id]:
                vf_value = 0
            else:
                # Build agent_id list for a shared policy
                shared_policy_agent_ids = [
                    agent_name
                    for agent_name in trainable_observations[episode.env_id]
                    if agent_name in policy_id_map[policy_id] and not bool(agent_termination_values[episode.env_id].get(agent_name))
                ]
                # Determine index into vf_tensor given agent_id in shared_policy list
                if agent_id in shared_policy_agent_ids:
                    index = shared_policy_agent_ids.index(agent_id)
                    vf_value = vf_tensor[index]

        output_dict[agent_id] = vf_value

    return output_dict
