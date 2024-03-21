"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from collections.abc import Mapping

from corl.episode_parameter_providers import EpisodeParameterProviderValidator
from corl.episode_parameter_providers.base_network_epp import BaseNetworkParameterProvider, NetworkedParameterProviderValidator
from corl.evaluation.connection.base_epp_update import get_updates
from corl.evaluation.runners.inference.protocol.simple_epp_update import SimpleEppUpdate
from corl.libraries.parameters import OverridableParameterWrapper, ParameterValidator
from corl.libraries.units import Quantity, corl_quantity


class SimpleNetworkEppValidator(NetworkedParameterProviderValidator[SimpleEppUpdate]):
    @property
    def prefix(self) -> str:
        if self.epp_variety == "environment":
            return "environment"
        if not self.agent_id:
            raise RuntimeError("Sanity Check Fail - 'agent_id' should be valid")
        return self.agent_id


class SimpleNetworkEpp(BaseNetworkParameterProvider[SimpleEppUpdate]):
    def __init__(self, **kwargs):
        self.config: SimpleNetworkEppValidator  # type: ignore
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[EpisodeParameterProviderValidator]:
        return SimpleNetworkEppValidator

    def _do_update_params(
        self, params: Mapping[tuple[str, ...], OverridableParameterWrapper], updates: SimpleEppUpdate
    ) -> Mapping[tuple[str, ...], OverridableParameterWrapper]:
        update_dict = get_updates(updates, self.config.prefix)

        def try_get_units(param: OverridableParameterWrapper) -> str:
            if isinstance(param.config, ParameterValidator):
                return param.config.units
            return ""

        for key, value in update_dict.items():
            if key not in params:
                raise RuntimeError(f"Parameter '{key}' not found in params")

            if isinstance(value, str):
                params[key].override_value = value
            else:
                param_units = try_get_units(params[key])
                if isinstance(value, Quantity):
                    params[key].override_value = value.to(param_units)
                else:
                    params[key].override_value = corl_quantity()(value=value, units=param_units)

        return params
