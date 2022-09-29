"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from typing import Any, Dict

from pydantic import BaseModel, PyObject


class Factory(BaseModel):
    """Factory class to allow subclass creation with pydantic

    TODO: Create a simple motivating example
    """
    type: PyObject
    config: Dict[str, Any] = {}

    def build(self, **kwargs):
        """Build the object contained within this factory."""
        return self.type(**self.config, **kwargs)

    @classmethod
    def resolve_factory(cls, v):
        """Validator for converting a factory into the built object.

        Usage in a pydantic model:
        resolve_factory = validator('name', pre=True, allow_reuse=True)(Factory.resolve_factory)
        """
        try:
            v['type']
        except (TypeError, KeyError):
            # Not something that should be built with the factory
            return v
        else:
            factory = cls(**v)
            return factory.build()
