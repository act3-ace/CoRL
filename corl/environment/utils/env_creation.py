"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import typing

from corl.agents.base_agent import AgentParseInfo, BaseAgent, PlatformParseInfo
from corl.episode_parameter_providers import EpisodeParameterProvider
from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_parts import BasePlatformPart


def create_agent_sim_configs(
    agents: dict[str, AgentParseInfo],
    agent_platforms: dict[str, PlatformParseInfo],
    sim_class: typing.Callable,
    epp_registry: dict[str, EpisodeParameterProvider],
    *,
    multiple_workers: bool = False,
) -> tuple[dict[str, BaseAgent], dict]:
    """
    Creates  dictionary of agent configs used byt the simmulator

    Parameters
    ----------
    agents: dictionary of agent names and configs
    sim_class: simulator class used by environment
    avail_platforms: available platforms in the simulator

    Returns
    -------
    None
    """
    sim_agent_configs = {}
    agent_dict = {}
    platform_part_dict: dict[str, list[BasePlatformPart]] = {}
    platform_cfg_dict: dict[str, list[dict]] = {}

    for platform_name in agent_platforms:
        platform_part_dict[platform_name] = []
        platform_cfg_dict[platform_name] = []

    for agent_name, agent_configs in agents.items():
        agent_class = agent_configs.class_config.agent(
            **agent_configs.class_config.config,
            epp=epp_registry[agent_name],
            agent_name=agent_name,
            platform_names=agent_configs.platform_names,
            multiple_workers=multiple_workers,
        )
        agent_dict[agent_name] = agent_class

        agent_platforms_dict = {
            platform_name: PluginLibrary.get_platform_from_sim_and_config(
                sim_class=sim_class, config=agent_platforms[platform_name]  # type: ignore[arg-type]
            )
            for platform_name in agent_configs.platform_names
        }
        partial_agent_part_dict = agent_class.get_platform_parts(sim_class, agent_platforms_dict)

        for platform_name in agent_configs.platform_names:
            for cls, cfg in partial_agent_part_dict[platform_name]:
                for i, part_cls in enumerate(platform_part_dict[platform_name]):
                    if cls == part_cls and platform_cfg_dict[platform_name][i] == cfg:
                        break
                else:
                    platform_part_dict[platform_name].append(cls)
                    platform_cfg_dict[platform_name].append(cfg)

    for platform_name, platform_config in agent_platforms.items():
        sim_agent_configs[platform_name] = {
            "platform_config": platform_config,
            "parts_list": list(zip(platform_part_dict[platform_name], platform_cfg_dict[platform_name])),
        }

    return agent_dict, sim_agent_configs
