"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import typing


def cleanup(fn: typing.Callable):
    """Returns an object that will call fn when it goes out of scope"""

    class ScopedDestructor:
        """calls fn when this goes out of scope"""

        def __init__(self):
            self.abort = False
            self.clean = False

        def __del__(self):
            self.cleanup()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.cleanup()

        def cleanup(self):
            if not self.abort and not self.clean:
                self.clean = True
                fn()

    return ScopedDestructor()
