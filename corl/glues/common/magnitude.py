"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Wrapper glue which returns the magnitude of its wrapped glue as an observation.

Author: Jamie Cunningham
"""
from collections import OrderedDict
from functools import cached_property

import numpy as np

from corl.glues.base_wrapper import BaseWrapperGlue, BaseWrapperGlueValidator
from corl.glues.common.observe_sensor import ObserveSensor
from corl.libraries.property import BoxProp, DictProp
from corl.libraries.units import Quantity, corl_quantity


class MagnitudeGlueValidator(BaseWrapperGlueValidator):
    """
    wrapped_glue: ObserveSensor
        wrapped observe sensor
    """

    wrapped: ObserveSensor


class MagnitudeGlue(BaseWrapperGlue):
    """
    Returns the magnitude of the wrapped glue's returned observation.
    """

    class Fields:
        """
        field data
        """

        MAG = "mag"

    def __init__(self, **kwargs) -> None:
        self.config: MagnitudeGlueValidator
        super().__init__(**kwargs)

        self._uname = f"{self.glue().get_unique_name()}_Magnitude"

        wrapped_prop = self.glue().observation_prop
        if not wrapped_prop or not isinstance(wrapped_prop, DictProp):
            raise RecursionError(
                "the glue the magnitude glue is wrapping does not have a valid obs space"
                "This glue expects the wrapped glue to provide a obs space and that space"
                " must be a Dict space"
            )

        child_prop = wrapped_prop.spaces[self.config.wrapped.Fields.DIRECT_OBSERVATION]
        assert isinstance(child_prop, BoxProp)
        self._prop = DictProp(
            spaces={
                self.Fields.MAG: BoxProp(
                    low=[0.0],
                    high=[np.maximum(np.linalg.norm(child_prop.low), np.linalg.norm(child_prop.high))],
                    shape=(1,),
                    dtype=np.dtype("float32"),
                    unit="dimensionless",
                )
            }
        )

    @staticmethod
    def get_validator() -> type[MagnitudeGlueValidator]:
        """returns the validator for this class

        Returns:
            MagnitudeGlueValidator -- A pydantic validator to be used to validate kwargs
        """
        return MagnitudeGlueValidator

    def get_unique_name(self) -> str:
        return self._uname

    @cached_property
    def observation_prop(self):
        return self._prop

    def get_observation(self, other_obs: OrderedDict, obs_space, obs_units):
        obs: Quantity = self.glue().get_observation(other_obs, obs_space, obs_units)  # type: ignore
        if isinstance(obs, dict):
            obs = obs[self.config.wrapped.Fields.DIRECT_OBSERVATION]
        else:
            raise RuntimeError("This glue is not capable of handing wrapped obs that are not a dict")
        d = OrderedDict()
        d[self.Fields.MAG] = corl_quantity()(np.array([np.linalg.norm(obs.m)], dtype=np.float32), "dimensionless")
        return d
