"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from collections import OrderedDict
from collections.abc import Callable

import numpy as np

ObsType = np.ndarray | tuple | dict


def mutate_observations(observations: OrderedDict, mutate_fn: Callable[[str, str, ObsType], ObsType]) -> OrderedDict:
    """
    Mutates observations according to mutate_fn
    if the MUTATED_OBSERVATION is not None, it will be included in the output

    Parameters
    ----------
    observations:
        An nested dictionary: observations[AGENT_ID][OBSERVATION_NAME] -> OBSERVATION
    mutate_fn(AGENT_ID: str, OBSERVATION_NAME: str, OBSERVATION: ObsType): MUTATED_OBSERVATION: ObsType
        Returns
        -------
            The mutated observation

    Returns
    -------
    OrderedDict:
        the mutated observation samples
    """
    mutated_observation_dict: OrderedDict = OrderedDict()

    for agent_id, obs_dict in observations.items():
        for obs_name, obs in obs_dict.items():
            mutated_obs = mutate_fn(agent_id, obs_name, obs)

            if mutated_obs is not None:
                if agent_id not in mutated_observation_dict:
                    mutated_observation_dict[agent_id] = OrderedDict()

                mutated_observation_dict[agent_id][obs_name] = mutated_obs

    return mutated_observation_dict


def filter_observations(observations: OrderedDict, filter_fn: Callable[[str, str, ObsType], bool]) -> OrderedDict:
    """
    Filters observations that don't match filter_fn

    Parameters
    ----------
    observations:
        An nested dictionary: observations[AGENT_ID][OBSERVATION_NAME] -> OBSERVATION
    filter_fn(AGENT_ID: str, OBSERVATION_NAME: str, OBSERVATION: ObsType): bool
        Returns
        -------
            true:  the observation will be included in the output
            false: the observation will not be included in the output

    Returns
    -------
    OrderedDict:
        the filtered observation samples
    """
    return mutate_observations(
        observations, lambda agent_id, obs_name, obs: obs if filter_fn(agent_id, obs_name, obs) else None  # type: ignore
    )
