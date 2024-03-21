from corl.agents.base_agent import NonTrainableBaseAgent


class NoOpAgent(NonTrainableBaseAgent):
    """
    Agent that does not send actions
    """

    def create_action(self, observation):  # noqa: PLR6301
        return {}
