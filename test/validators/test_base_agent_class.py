# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

from corl.agents.base_agent import AgentParseBase
from corl.parsers.yaml_loader import load_file


def test_basic_agent_parse():
    # This test goes straight from yml to an agent class object
    config = load_file("config/tasks/gymnasium/agents/gymnasium_agent.yml")

    parsed_agent = AgentParseBase(**config)
    parsed_agent.agent(**parsed_agent.config, agent_name="foobar", platform_names=["blue0"])


def test_basic_platform_parse():
    load_file("config/tasks/gymnasium/platforms/gymnasium_platform.yml")
