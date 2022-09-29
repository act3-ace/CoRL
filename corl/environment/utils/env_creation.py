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


def create_agent_sim_configs(
    agents: typing.Dict[str, AgentParseInfo],
    agent_platforms: typing.Dict[str, PlatformParseInfo],
    sim_class: typing.Callable,
    avail_platforms: typing.Callable,
    epp_registry: typing.Dict[str, EpisodeParameterProvider],
    *,
    multiple_workers: bool = False
) -> typing.Tuple[typing.Dict[str, BaseAgent], dict]:
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

    for platform_name, platform_config in agent_platforms.items():
        agent_platform = avail_platforms.ParseFromNameModel(platform_config)  # type: ignore

        agent_part_cls: typing.List = []
        agent_part_cfg: typing.List = []
        for agent_name, agent_configs in agents.items():
            if agent_configs.platform_name != platform_name:
                continue

            agent_class = agent_configs.class_config.agent(
                **agent_configs.class_config.config,
                epp=epp_registry[agent_name],
                agent_name=agent_name,
                platform_name=platform_name,
                multiple_workers=multiple_workers
            )
            agent_dict[agent_name] = agent_class
            partial_agent_part_list = agent_class.get_platform_parts(sim_class, agent_platform)

            for (cls, cfg) in partial_agent_part_list:
                unique = True
                for i, part_cls in enumerate(agent_part_cls):
                    if cls == part_cls and agent_part_cfg[i] == cfg:
                        unique = False
                        break

                if unique:
                    agent_part_cls.append(cls)
                    agent_part_cfg.append(cfg)

        agent_part_list = list(zip(agent_part_cls, agent_part_cfg))

        sim_agent_configs[platform_name] = {
            "platform_config": platform_config,
            "parts_list": agent_part_list,
        }

    return agent_dict, sim_agent_configs
