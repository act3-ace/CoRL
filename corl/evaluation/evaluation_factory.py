"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from pydantic import BaseModel, Field, ImportString


class FactoryLoader(BaseModel):
    """Object factory which matches jsonargparse interface"""

    class_path: ImportString
    init_args: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)

    @classmethod
    def resolve_factory(cls, value, info):
        if isinstance(value, dict):
            factory = FactoryLoader(**value)
            if issubclass(factory.class_path, BaseModel):
                return factory.class_path.model_validate({**factory.init_args, **factory.config}, context=info.context)
            return factory.class_path(**factory.init_args, **factory.config)

        return value
