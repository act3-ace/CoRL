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
from corl.glues.common.controller_glue import ControllerGlue


class RandomActionAgent(NonTrainableBaseAgent):
    """
    Agent that randomly samples control spce
    """

    def create_action(self, observation):
        """
        Creates a random action based on control space.

        Parameters
        ----------
        observation (object): agent's observation of the current environment

        Returns
        -------
        OrderedDict[str: Any]
        A dictionary of unnormalized actions applied to glues in the form {glue_name: raw_action}
        """
        control_dict = {}

        for glue_name, glue in self.agent_glue_dict.items():
            if isinstance(glue, ControllerGlue):
                control_dict[glue_name] = glue.action_space.sample()
        return control_dict
