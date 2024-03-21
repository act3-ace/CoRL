"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel

_context_stack = ContextVar[list[dict[str, Any]]]("_context_stack", default=[])


@contextmanager
def add_context(value: dict[str, Any]) -> Iterator[None]:
    ctx: list[dict[str, Any]] = []
    ctx.extend(_context_stack.get())
    ctx.append(value)

    token = _context_stack.set(ctx)
    try:
        yield
    finally:
        _context_stack.reset(token)


def get_current_context() -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    for d in _context_stack.get():
        ctx.update(d)
    return ctx


class BaseModelWithContext(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        __pydantic_self__.__pydantic_validator__.validate_python(
            data,
            self_instance=__pydantic_self__,
            context=get_current_context(),
        )
