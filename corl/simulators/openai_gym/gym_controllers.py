"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Provides controllers to pass actions to wrapped OpenAI Gym Environments
"""

import numbers
import typing

import gym
import numpy as np

from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.property import BoxProp, DiscreteProp
from corl.simulators.base_parts import BaseController
from corl.simulators.openai_gym.gym_available_platforms import OpenAIGymAvailablePlatformTypes
from corl.simulators.openai_gym.gym_simulator import OpenAIGymSimulator


class OpenAIGymMainController(BaseController):
    """
    GymController implementation for passing actions to wrapped OpenAI Gym Environments
    """

    def __init__(self, parent_platform, config=None):

        act_space = parent_platform.action_space
        cont_prop: typing.Union[DiscreteProp, BoxProp]

        if isinstance(act_space, gym.spaces.Discrete):

            class DiscreteGymProp(DiscreteProp):
                """
                DiscreteGymProp can be updated via config and valdidate by pydantic
                """
                name: str = "gym controller"
                unit: str = "None"
                n: int = act_space.n
                description: str = "gym env action space"

            cont_prop = DiscreteGymProp

        elif isinstance(act_space, gym.spaces.Box):

            class BoxGymProp(BoxProp):
                """
                BoxGymProp can be updated via config and valdidate by pydantic
                """
                name: str = "gym controller"
                low: typing.List[float] = act_space.low.tolist()
                high: typing.List[float] = act_space.high.tolist()
                dtype: np.dtype = act_space.dtype
                unit: typing.List[str] = ["None"] * len(act_space.low)
                description: str = "gym env action space"

            cont_prop = BoxGymProp
        else:
            raise RuntimeError(f"This controller does not currently know how to handle a {type(act_space)} action space")

        super().__init__(parent_platform=parent_platform, config=config, property_class=cont_prop)

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return {"main_controller"}

    def apply_control(self, control: np.ndarray) -> None:
        self.parent_platform.save_action_to_platform(control)

    def get_applied_control(self) -> typing.Union[np.ndarray, numbers.Number]:
        return self.parent_platform.get_applied_action()


PluginLibrary.AddClassToGroup(
    OpenAIGymMainController, "Controller_Gym", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)


class OpenAIGymDuplicateController(OpenAIGymMainController):
    """
    GymController implementation for passing actions to wrapped OpenAI Gym Environments
    """

    @property
    def exclusiveness(self) -> typing.Set[str]:
        """Return exclusiveness"""
        return set()


PluginLibrary.AddClassToGroup(
    OpenAIGymDuplicateController,
    "Controller_Duplicate_Gym", {
        "simulator": OpenAIGymSimulator, "platform_type": OpenAIGymAvailablePlatformTypes.MAIN
    }
)
