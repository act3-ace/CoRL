"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from typing import Any

from pydantic import BaseModel, ImportString, ValidationInfo


def _construct(type_, info, **kwargs):
    if info and info.context:
        if issubclass(type_, BaseModel):
            return type_.model_validate(kwargs, context=info.context)

        try:
            validated_args = type_.get_validator().model_validate(kwargs, context=info.context)
            return type_(**dict(validated_args))
        except AttributeError:
            pass

    return type_(**kwargs)


class Factory(BaseModel):
    """Factory class to allow subclass creation with pydantic

    TODO: Create a simple motivating example
    """

    type: ImportString  # noqa: A003
    wrapped: dict[str, "Factory"] = {}
    config: dict[str, Any] = {}

    def build(self, info: ValidationInfo | None = None, **kwargs):
        """Build the object contained within this factory."""

        ctor_kwargs = {**self.config, **kwargs}
        if self.wrapped:
            ctor_kwargs["wrapped"] = {k: v.build(**kwargs) for k, v in self.wrapped.items()}

        return _construct(self.type, info, **ctor_kwargs)

    @classmethod
    def resolve_factory(cls, v, info):
        """Validator for converting a factory into the built object.

        Usage in a pydantic model:
        resolve_factory = field_validator('name')(Factory.resolve_factory)
        """
        try:
            v["type"]
        except (TypeError, KeyError):
            # Not something that should be built with the factory
            return v
        else:
            factory = cls(**v)
            return factory.build(info=info)
