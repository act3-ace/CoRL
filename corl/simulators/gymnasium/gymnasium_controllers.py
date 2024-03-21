"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Provides controllers to pass actions to wrapped  Gymnasium Environments
"""

import gymnasium
import numpy as np

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp, DiscreteProp
from corl.libraries.units import Quantity, corl_get_ureg
from corl.simulators.base_parts import BaseController
from corl.simulators.gymnasium.gymnasium_available_platforms import GymnasiumAvailablePlatformTypeMain
from corl.simulators.gymnasium.gymnasium_simulator import GymnasiumSimulator


class GymnasiumMainController(BaseController):
    """
    GymnasiumController implementation for passing actions to wrapped  Gymnasium Environments
    """

    def __init__(self, parent_platform, config=None) -> None:
        act_space = parent_platform.action_space
        cont_prop: type[DiscreteProp] | type[BoxProp]

        if isinstance(act_space, gymnasium.spaces.Discrete):

            class DiscreteGymnasiumProp(DiscreteProp):
                """
                DiscreteGymnasiumProp can be updated via config and validate by pydantic
                """

                name: str = "gymnasium controller"
                unit: str = "dimensionless"
                n: int = act_space.n
                description: str = "gymnasium env action space"

            cont_prop = DiscreteGymnasiumProp

        elif isinstance(act_space, gymnasium.spaces.Box):

            class BoxGymnasiumProp(BoxProp):
                """
                BoxGymnasiumProp can be updated via config and validate by pydantic
                """

                name: str = "gymnasium controller"
                low: list[float] = act_space.low.tolist()
                high: list[float] = act_space.high.tolist()
                dtype: np.dtype = act_space.dtype
                unit: str = "dimensionless"
                description: str = "gymnasium env action space"

            cont_prop = BoxGymnasiumProp
        else:
            raise RuntimeError(f"This controller does not currently know how to handle a {type(act_space)} action space")

        super().__init__(parent_platform=parent_platform, config=config, property_class=cont_prop)
        new_action = corl_get_ureg().Quantity(self.control_properties.create_space().sample(), "dimensionless")
        parent_platform.save_action_to_platform(new_action)

    def apply_control(self, control) -> None:
        assert isinstance(control, Quantity)
        self.parent_platform.save_action_to_platform(control)

    def get_applied_control(self) -> Quantity:
        return self.parent_platform.get_applied_action()


PluginLibrary.AddClassToGroup(
    GymnasiumMainController, "Controller_Gymnasium", {"simulator": GymnasiumSimulator, "platform_type": GymnasiumAvailablePlatformTypeMain}
)
