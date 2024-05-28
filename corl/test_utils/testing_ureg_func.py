# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
from collections.abc import Callable

from corl.libraries.units import UnitRegistryConfiguration, corl_set_ureg


def ureg_setup_func(programatic_defines: list[Callable] | None = None):
    if programatic_defines is None:
        programatic_defines = []
    ureg = UnitRegistryConfiguration(programatic_defines=programatic_defines).create_registry_from_args()
    corl_set_ureg(ureg)
