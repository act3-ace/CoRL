"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------

A set of standard config updates that can be used.
"""
from corl.evaluation.runners.section_factories.plugins.config_updater import ConfigUpdate


class DoNothingConfigUpdate(ConfigUpdate):
    """
    A config update that does nothing. Used as a placeholder/default when no config update is required.
    """

    def update(self, config):  # noqa: PLR6301
        return config
