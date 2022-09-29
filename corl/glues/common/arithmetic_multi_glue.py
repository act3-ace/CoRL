"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
ArithmeticMultiGlue implementation
"""
import typing
from collections import OrderedDict
from functools import lru_cache

import gym
import numpy as np

from corl.glues.base_multi_wrapper import BaseMultiWrapperGlue, BaseMultiWrapperGlueValidator
from corl.glues.common.target_value import LimitConfigValidator


class ArithmeticMultiGlueValidator(BaseMultiWrapperGlueValidator):
    """
    mode: what arithmetic operation to run on the output of the wrapped glues
    limit: the expected limit for this glue
    """
    mode: typing.Literal["sum", "sub", "mult", "div"] = "sum"
    limit: LimitConfigValidator


# TODO: Support for "sub", "mult", "div" is broken
class ArithmeticMultiGlue(BaseMultiWrapperGlue):
    """
    ArithmeticMultiGlue takes in a list of wrapped glues and performs some
    arithmetic operation on their output
    """

    def __init__(self, **kwargs) -> None:
        self.config: ArithmeticMultiGlueValidator
        super().__init__(**kwargs)
        self.operator = np.sum
        # if self.config.mode == "sub":
        #     self.operator = np.subtract
        # elif self.config.mode == "mult":
        #     self.operator = np.multiply
        # elif self.config.mode == "div":
        #     self.operator = np.divide
        self.field_names = []
        for glue in self.glues():
            space = glue.observation_space()
            if len(space.spaces) > 1:
                raise RuntimeError("ArithmeticMultiGlue can only wrap a glue with one output")
            self.field_names.append(list(space.spaces.keys())[0])

    class Fields:
        """
        Field data
        """
        RESULT = "result"

    @property
    def get_validator(self) -> typing.Type[ArithmeticMultiGlueValidator]:
        return ArithmeticMultiGlueValidator

    @lru_cache(maxsize=1)
    def get_unique_name(self):
        """Class method that retreives the unique name for the glue instance
        """
        tmp = [glue.get_unique_name() for glue in self.glues()]
        if any(tmp_str is None for tmp_str in tmp):
            return None
        wrapped_glue_names = "".join(tmp)
        return wrapped_glue_names + self.config.mode

    def invalid_value(self) -> OrderedDict:
        """When invalid return a value of 0

        TODO: this may need to be self.min in the case that the minimum is larger than 0 (i.e. a harddeck)

        Returns:
            OrderedDict -- Dictionary with <FIELD> entry containing 1D array
        """
        d = OrderedDict()
        d[f"{self.Fields.RESULT}"] = np.asarray(
            [(self.config.limit.maximum + self.config.limit.minimum) / 2], dtype=np.float32
        )  # type: ignore
        return d

    @lru_cache(maxsize=1)
    def observation_space(self):
        d = gym.spaces.dict.Dict()

        d.spaces[f"{self.Fields.RESULT}"] = gym.spaces.Box(
            self.config.limit.minimum, self.config.limit.maximum, shape=(1, ), dtype=np.float32
        )
        return d

    def get_observation(self):
        d = OrderedDict()
        tmp_output = [glue.get_observation()[field_name] for glue, field_name in zip(self.glues(), self.field_names)]
        d[self.Fields.RESULT] = np.array([self.operator(tmp_output)], dtype=np.float32)
        return d

    @lru_cache(maxsize=1)
    def action_space(self) -> gym.spaces.Space:
        return None

    def apply_action(self, action, observation):
        return None
