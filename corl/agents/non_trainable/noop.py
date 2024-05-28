# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
from corl.agents.base_agent import NonTrainableBaseAgent


class NoOpAgent(NonTrainableBaseAgent):
    """
    Agent that does not send actions
    """

    def create_action(self, observation):  # noqa: PLR6301
        return {}
