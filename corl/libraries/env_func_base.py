"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Functors are objects the that can be treated as though they are a function.
When to use functors?
- Functors are used when you want to hide/abstract the real implementation.
Let's say you want to call the different functions depending on the input
but you don't want the user code to make explicit calls to those different
functions. This is the ideal situation where functors can help.
- In this scenario, we can go for a functor which internally calls the most
suitable function depending on the input.
- Now if later, none of functions to be called increases, then it would be
just a simple change in the backend code without disturbing any of the user
code. Thus functors help in creating maintainable, decoupled and extendable
 codes.
"""


class EnvFuncBase:
    """base definition for env functions"""

    def reset(self):
        """Base reset function for items such as rewards and dones"""

    @property
    def name(self) -> str:
        """gets the name for the functor

        Returns
        -------
        str
            The name of the functor
        """
        return type(self).__name__
