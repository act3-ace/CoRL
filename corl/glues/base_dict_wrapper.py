"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from pydantic import ConfigDict

from corl.glues.base_glue import BaseAgentGlue, BaseAgentGlueValidator


class BaseDictWrapperGlueValidator(BaseAgentGlueValidator):
    """
    wrap_dict - A dict of the wrapped glues
    """

    wrapped: dict[str, BaseAgentGlue]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseDictWrapperGlue(BaseAgentGlue):
    """A base object that glues can inherit in order to "wrap" multiple glue instances, addressed by keys"""

    def __init__(self, **kwargs) -> None:
        self.config: BaseDictWrapperGlueValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[BaseDictWrapperGlueValidator]:
        return BaseDictWrapperGlueValidator

    def glues(self) -> dict[str, BaseAgentGlue]:
        """Get the wrapped glue instances dict"""
        return self.config.wrapped

    def set_agent_removed(self, agent_removed: bool = True) -> None:
        super().set_agent_removed(agent_removed)
        for glue in self.glues().values():
            glue.set_agent_removed()
