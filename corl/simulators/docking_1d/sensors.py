"""
This module contains implementations of Sensors that reside on the Docking1dPlatform.
"""
from corl.libraries.plugin_library import PluginLibrary
from corl.libraries.units import Quantity, corl_get_ureg
from corl.simulators.base_parts import BaseSensor
from corl.simulators.docking_1d.available_platforms import Docking1dAvailablePlatformType
from corl.simulators.docking_1d.properties import PositionProp, VelocityProp
from corl.simulators.docking_1d.simulator import Docking1dSimulator


class PositionSensor(BaseSensor):
    """
    A Sensor to measure the position of the associated Docking1dPlatform.
    """

    def __init__(self, parent_platform, config, measurement_properties=PositionProp):
        super().__init__(parent_platform=parent_platform, config=config, property_class=measurement_properties)

    def _calculate_measurement(self, state) -> Quantity:
        """
        get measurements from the sensor
        """
        return corl_get_ureg().Quantity(self.parent_platform.position.astype("float32"), "meter")


class VelocitySensor(BaseSensor):
    """
    A Sensor to measure the velocity of the associated Docking1dPlatform.
    """

    def __init__(self, parent_platform, config, measurement_properties=VelocityProp):
        super().__init__(parent_platform=parent_platform, config=config, property_class=measurement_properties)

    def _calculate_measurement(self, state) -> Quantity:
        """
        get measurements from the sensor
        """
        return corl_get_ureg().Quantity(self.parent_platform.velocity.astype("float32"), "meter / second")


# Register sensors with PluginLibrary. Requires a class, reference name, and a dict of Simulator class and platform
# type enum.

PluginLibrary.AddClassToGroup(
    PositionSensor, "1D_Sensor_Position", {"simulator": Docking1dSimulator, "platform_type": Docking1dAvailablePlatformType}
)

PluginLibrary.AddClassToGroup(
    VelocitySensor, "1D_Sensor_Velocity", {"simulator": Docking1dSimulator, "platform_type": Docking1dAvailablePlatformType}
)
