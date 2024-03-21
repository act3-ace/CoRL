"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from collections import OrderedDict


class ObsBuffer:
    """Wrapper for OBS functions"""

    def __init__(self) -> None:
        """Constructor"""
        self._buffer1: OrderedDict = OrderedDict()
        self._buffer2: OrderedDict = OrderedDict()
        self._index = 0

    def update_obs_pointer(self):
        """get the next observation pointer for next and current"""
        self._index += 1

    @property
    def observation(self) -> OrderedDict:
        """Every other call we update the pointer to the correct obs buffer
        Returns
        -------
        OrderedDict
            The observation dictionary
        """
        return self._buffer1 if self._index % 2 == 0 else self._buffer2

    @observation.setter
    def observation(self, data: OrderedDict) -> None:
        """sets the observation

        Parameters
        ----------
        data : OrderedDict
            The update data
        """
        if self._index % 2 == 0:
            # Environment.ObsBuffer.update_dict(self._buffer1, data)
            self._buffer1 = data
        else:
            # Environment.ObsBuffer.update_dict(self._buffer2, data)
            self._buffer2 = data

    @property
    def next_observation(self) -> OrderedDict:
        """Every other call we update the pointer to the correct next obs buffer
        Returns
        -------
        OrderedDict
            The next observation dictionary
        """
        return self._buffer2 if self._index % 2 == 0 else self._buffer1

    @next_observation.setter
    def next_observation(self, data: OrderedDict) -> None:
        """sets the next observation

        Parameters
        ----------
        data : OrderedDict
            The update data
        """
        if self._index % 2 == 0:
            # Environment.ObsBuffer.update_dict(self._buffer2, data)
            self._buffer2 = data
        else:
            # Environment.ObsBuffer.update_dict(self._buffer1, data)
            self._buffer1 = data
