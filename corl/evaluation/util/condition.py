"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import typing

from pydantic import BaseModel, PrivateAttr

from corl.libraries.factory import Factory


class Condition(BaseModel):
    """Creates boolean condition logic from a lhs and a operator

    Results in a boolean function which the lhs is to be provided
    """

    operator: str
    lhs: int | (float | dict)

    _func: typing.Callable[[typing.Any], bool] = PrivateAttr()
    """
    This lambda is private, it is built in the constructor based on non private members
    """

    def __init__(self, **data):
        super().__init__(**data)

        # build the lhs
        if isinstance(self.lhs, float | int):
            lhs = self.lhs
        elif isinstance(self.lhs, dict):
            if "functor" in self.lhs:
                lhs = Factory(type=self.lhs["functor"], config=self.lhs["config"]).build()
            else:
                raise RuntimeError("Unknown how to build lhs")
        else:
            raise RuntimeError(f"Unknown type {type(self.lhs)} on the lhs of CriteriaRate")

        # Construct the criteria_func method
        if self.operator == "==":
            self._func = lambda x: x == lhs
        elif self.operator == "<":
            self._func = lambda x: x < lhs
        elif self.operator == ">":
            self._func = lambda x: x > lhs
        else:
            raise RuntimeError(f"Unknown Operator: {self.operator}")

    def func(self, rhs: typing.Any) -> bool:
        """Execute the function wrapped up in this conditional

        Args:
            rhs (typing.Any): Right hand side

        Returns:
            bool: Result of condition
        """
        return self._func(rhs)


class NamedCondition(BaseModel):
    """Wraps a Condition with a name"""

    type: str  # noqa: A003
    condition: Condition
