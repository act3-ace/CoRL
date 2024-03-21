from collections.abc import Callable

from corl.libraries.units import UnitRegistryConfiguration, corl_set_ureg


def ureg_setup_func(programatic_defines: list[Callable] | None = None):
    if programatic_defines is None:
        programatic_defines = []
    ureg = UnitRegistryConfiguration(programatic_defines=programatic_defines).create_registry_from_args()
    corl_set_ureg(ureg)
