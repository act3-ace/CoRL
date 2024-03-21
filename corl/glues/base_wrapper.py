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


class BaseWrapperGlueValidator(BaseAgentGlueValidator):
    """
    wrapped - the wrapped glue instance
    """

    wrapped: BaseAgentGlue
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseWrapperGlue(BaseAgentGlue):
    """A base object that glues can inherit in order to "wrap" a single glue instance"""

    def __init__(self, **kwargs) -> None:
        self.config: BaseWrapperGlueValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[BaseWrapperGlueValidator]:
        return BaseWrapperGlueValidator

    def glue(self) -> BaseAgentGlue:
        """Get the wrapped glue instance"""
        return self.config.wrapped

    def set_agent_removed(self, agent_removed: bool = True) -> None:
        super().set_agent_removed(agent_removed)
        self.glue().set_agent_removed(agent_removed)
