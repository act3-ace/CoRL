"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
state Dict - Leverage https://github.com/ramazanpolat/StateDict -
"""
import copy
from collections import OrderedDict

DICT_RESERVED_KEYS = vars(OrderedDict).keys()


class StateDict(OrderedDict):
    """[summary]

    Parameters
    ----------
    OrderedDict : [type]
        [description]
    """

    def __init__(self, *args, **kwargs):
        self._recurse = kwargs.pop("recurse", True)
        super().__init__(*args, **kwargs)
        if self._recurse:
            super().__init__(StateDict.recursive_attrdict(self))

    @staticmethod
    def recursive_attrdict(obj):
        """Walks a simple data structure, converting dictionary to StateDict.
        Supports lists, tuples, and dictionaries.
        """
        ret = obj
        if isinstance(obj, dict):
            if issubclass(type(obj), StateDict):
                for k, v in obj.items():
                    obj[k] = StateDict.recursive_attrdict(v)

                ret = obj
            else:
                ret = StateDict(
                    {str(k): StateDict.recursive_attrdict(v) for (k, v) in obj.items()},
                    recurse=False,
                )
        elif isinstance(obj, list):
            ret = [StateDict.recursive_attrdict(i) for i in obj]
        elif isinstance(obj, tuple):
            ret = tuple(StateDict.recursive_attrdict(i) for i in obj)

        return ret

    def __setattr__(self, name, value):
        if isinstance(name, str) and name[0] != "_":
            self.__setitem__(name, value)
        super().__setattr__(name, value)

    def __getattr__(self, name):
        _item = self.get(name)  # __getitem__(key) if key in self else None

        if name not in self and _item is None and isinstance(name, str) and name[0] != "_":
            # attempt to ignore builtins and private attributes
            raise KeyError(f"{name}")

        return _item

    def __deepcopy__(self, memo):
        return StateDict(copy.deepcopy(dict(self)), recurse=self._recurse)

    def __delattr__(self, name):
        self.__delitem__(name)
        super().__delattr__(name)

    def __dir__(self):
        return list(super().__dir__()) + [str(k) for k in self.keys()]

    def keys(self):  # noqa: PLR6301
        return OrderedDict(sorted(super().items())).keys()

    def values(self):  # noqa: PLR6301
        return OrderedDict(sorted(super().items())).values()

    def items(self):  # noqa: PLR6301
        return OrderedDict(sorted(super().items())).items()

    @staticmethod
    def merge(dl):
        """[summary]

        Parameters
        ----------
        dl : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        r = {}
        for d in dl:
            r = StateDict(sorted(StateDict.merge_two(r, d)))
        return StateDict(sorted(r.items()))

    @staticmethod
    def stack_values(dl):
        """[summary]

        Parameters
        ----------
        dl : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        ak = set().union(*dl)
        r = {k: [] for k in ak}
        for d in dl:
            for k, v in d.items():
                r[k].append(v)
        return r

    @staticmethod
    def merge_two(d1, d2, replace=True):
        """Merges two dicts

        Parameters
        ----------
        d1 : [type]
            [description]
        d2 : [type]
            [description]
        replace : bool, optional
            [description], by default True

        Yields
        ------
        [type]
            [description]

        Raises
        ------
        an
            [description]
        """
        for k in set(d1.keys()).union(d2.keys()):
            if k in d1 and k in d2:
                if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                    yield (k, StateDict(sorted(StateDict.merge_two(d1[k], d2[k]))))
                elif replace:
                    # replace the value on the first with the value in the second
                    yield (k, d2[k])
                else:
                    # maybe raise an exception, but default is to proceed without whining
                    pass
            elif k in d1:
                yield (k, d1[k])
            else:
                yield (k, d2[k])

    def to_dict(self) -> dict:
        """Converts to a dictionary

        Returns
        -------
        dict
            the dict form
        """
        temp = copy.deepcopy(self)
        for k, v in temp.items():
            if isinstance(v, StateDict):
                temp[k] = v.to_dict()
        return temp
