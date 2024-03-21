"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Environment Dict Module
"""
import abc
import copy
import logging
import typing
import warnings
from collections import OrderedDict

import numpy as np

from corl.libraries.state_dict import StateDict


class Callback:
    """
    Callback provides basic callback processing for all reward and done functions
    """

    def __init__(self, funcs: typing.Sequence[typing.Callable] | None = None) -> None:
        """
        __init__ constructor

        Parameters
        ----------
        funcs : typing.List[typing.Callable], optional
            List of callale functions, by default None
        """
        self._process_callbacks: list[typing.Callable] = []
        self._logger = logging.getLogger(Callback.__name__)
        if funcs:
            self.register_funcs(funcs)

    def register_func(self, func: typing.Callable) -> None:
        """
        register_func registers a function to the list of valid functions

        Parameters
        ----------
        func : typing.Callable
            The callable function
        """
        # MTB - 10/15/2020 - There seems to be an issue with this when combining multiple reward sets
        # if func is a callable the expression func in self._process_callbacks will always return True
        if isinstance(func, Callback) or func not in self._process_callbacks:
            self._process_callbacks.append(func)
        else:
            warnings.warn("Ignoring a duplicate callback given")

    def register_funcs(self, funcs: typing.Sequence[typing.Callable] | None):
        """
        register_func registers a list of functions to the list of valid functions

        Parameters
        ----------
        func : typing.Callable
            The callable function
        """
        for func in funcs or []:
            self.register_func(func)

    def unregister_func(self, func: typing.Callable) -> None:
        """
        unregister callbacks from processing

        Parameters
        ----------
        key : str
            The callback string to remove
        """
        if func in self._process_callbacks:
            self._process_callbacks.remove(func)

    def unregister_funcs(self) -> None:
        """
        unregister callbacks from processing
        """
        self._process_callbacks.clear()

    def reset_funcs(self):
        """
        reset_funcs func is a callable then attempt to reset its state.

        Parameters
        ----------
        key : str
            The callback string to remove
        """
        for func in self._process_callbacks:
            reset_op = getattr(func, "reset", None)
            if callable(reset_op):
                func.reset()

    @property
    def process_callbacks(self) -> list[typing.Callable]:
        """
        process_callbacks gets the current callbacks

        Returns
        -------
        typing.List[typing.Callable]
            Current list of callbacks
        """
        return self._process_callbacks


class EnvDict(StateDict, Callback):
    """[summary]

    Parameters
    ----------
    StateDict : [type]
        [description]
    Callback : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # list of items to exclude when generating the keys, items, values
    EXCLUDE_KEYS = [
        "_default_kwargs",
        "_reduce_fn",
        "_reduce_fn_kwargs",
        "_processing_funcs",
        "_recurse",
    ]

    class DuplicateName(RuntimeError):
        """Exception class for callbacks with duplicate names"""

    def __init__(
        self,
        processing_funcs: typing.Sequence[typing.Callable] | None = None,
        reduce_fn: typing.Callable | None = None,
        reduce_fn_kwargs=None,
        **kwargs,
    ) -> None:
        """
        __init__ environment dictionary constructor

        Parameters
        ----------
        processing_funcs : typing.List[typing.Callable], optional
            List of functions to call by the environment, by default None
        reduce_fn : typing.Callable, optional
            function used to reduce the results, by default None
        reduce_fn_kwargs : [type], optional
            [description], by default None
        """
        self._default_kwargs = None
        self._reduce_fn = reduce_fn
        self._reduce_fn_kwargs = reduce_fn_kwargs or {}
        self._set_default_kwargs(kwargs)
        StateDict.__init__(self, **kwargs)
        Callback.__init__(self, processing_funcs)

    def __call__(self, *args, **kwargs) -> tuple[OrderedDict, OrderedDict]:
        """
        __call__ Callable function for the environment dictionary type

        Returns
        -------
        typing.Tuple[OrderedDict, OrderedDict]
            The reduced rewards and theret information
        """
        r = [
            self._default_kwargs,
        ]
        ret_info: OrderedDict = OrderedDict()

        def merge(source, destination):
            """
            run me with nosetests --with-doctest file.py

            >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
            >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
            >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
            True
            """
            for key, value in source.items():
                if isinstance(value, dict):
                    # get node or create one
                    node = destination.setdefault(key, {})
                    merge(value, node)
                else:
                    destination[key] = value

            return destination

        for func in self._filtered_process_callbacks:
            if isinstance(func, OrderedDict):
                ret = func(*args, **kwargs)
                rew, info = ret
                r.append(rew)
                ret_info = merge(ret_info, info)

        for func in self._filtered_process_callbacks:
            if not isinstance(func, OrderedDict):
                ret = func(*args, **kwargs)
                # single value
                r.append(copy.deepcopy(ret))

                try:
                    name = func.__name__
                except:  # noqa: E722
                    name = func.name  # type: ignore

                # This only affects the info dictionary that is returned.  As the code below merges the output for all agents together,
                # the __all__ entry would be overwritten to only provide the information of the last agent, which could be confusing or
                # inaccurate.  Therefore, remove __all__ from the returned information.
                if "__all__" in ret:
                    del ret["__all__"]

                if name in ret_info:
                    common_keys = set(ret.keys()) & set(ret_info[name].keys())
                    if common_keys:
                        raise self.DuplicateName(f"{name} has common keys: {common_keys}")
                    ret_info[name].update(**ret)
                else:
                    ret_info[name] = ret

        # TODO link with reduce and ret info
        return self._reduce(r, **self._reduce_fn_kwargs), ret_info  # type: ignore

    @property
    def _filtered_process_callbacks(self) -> list[typing.Callable]:
        """Set of callbacks that have been filtered by subclass logic.

        Default implementation is all callbacks

        Returns
        -------
        typing.List[typing.Callable]
            Callbacks to apply
        """
        return self._process_callbacks

    @abc.abstractmethod
    def _reduce(self, r: typing.Callable, **kwargs):
        """
        _reduce user defined reduce function for processing

        Parameters
        ----------
        r : typing.Callable
            The reduce function to use
        """

    def _set_default_kwargs(self, kwargs):
        """
        _set_default_kwargs [summary]

        [extended_summary]

        Parameters
        ----------
        kwargs : [type]
            [description]
        """
        if self._default_kwargs is None:
            self._default_kwargs = copy.deepcopy(kwargs)

    def reset(self):
        """
        reset [summary]
        """
        # EnvDict.__init__(self, **self._default_kwargs)
        self.reset_funcs()

    def _filtered_self(self):
        tmp = {k: v for k, v in super().items() if k not in self.EXCLUDE_KEYS}
        return StateDict(tmp)

    def keys(self):
        return self._filtered_self().keys()

    def values(self):
        return self._filtered_self().values()

    def items(self):
        return self._filtered_self().items()

    def to_dict(self):
        return self._filtered_self().to_dict()

    @property
    def name(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return type(self).__name__

    @staticmethod
    def observation_deep_copy(d1, d2):
        """[summary]

        Parameters
        ----------
        d1 : [type]
            [description]
        d2 : [type]
            [description]
        """
        # d1 = pickle.loads(pickle.dumps(d2))
        for (k1, v1), (k2, v2) in zip(d1.items(), d2.items()):
            if isinstance(v1, dict):
                EnvDict.observation_deep_copy(v1, v2)
            else:
                d1[k1] = copy.deepcopy(d2[k2])


class DoneDict(EnvDict):
    """[summary]

    Parameters
    ----------
    EnvDict : [type]
        [description]
    """

    def __init__(
        self,
        processing_funcs: list[typing.Callable] | None = None,
        reduce_fn: typing.Callable | None = None,
        reduce_fn_kwargs=None,
        **kwargs,
    ) -> None:
        super().__init__(processing_funcs=processing_funcs, reduce_fn=reduce_fn, reduce_fn_kwargs=reduce_fn_kwargs, **kwargs)

        self._agent_filter: typing.Iterable[str] | None = None

    def __call__(self, *args, **kwargs) -> tuple[OrderedDict[typing.Any, typing.Any], OrderedDict[typing.Any, typing.Any]]:
        """
        __call__ Callable function for the done dictionary type

        Returns
        -------
        typing.Tuple[OrderedDict, OrderedDict]
            The done information
        """
        r = super().__call__(*args, **kwargs)

        # Check for bool type in return value
        for key, value in r[0].items():
            if isinstance(value, np.bool_):
                r[0][key] = bool(value)
            elif not isinstance(value, bool):
                raise TypeError(f"DoneDict __call__ return is not type bool for key: {key}")
        # remap from platform: {Done: {platform: bool}} -> platform: {Done: bool}
        tmp: OrderedDict[str, OrderedDict[str, bool]] = OrderedDict(
            [(platform_name, OrderedDict()) for platform_name in r[0] if platform_name != "__all__"]
        )
        for key0, value0 in r[1].items():
            for platform_name, platform_info in tmp.items():
                platform_info[key0] = bool(value0.get(platform_name, False))

        return (r[0], tmp)

    @property
    def _filtered_process_callbacks(self) -> list[typing.Callable]:
        if self._agent_filter is None:
            return super()._filtered_process_callbacks

        # Avoid circular import
        from corl.dones.done_func_base import DoneFuncBase

        return [x for x in self._process_callbacks if not isinstance(x, DoneFuncBase) or x.agent in self._agent_filter]

    def set_alive_agents(self, alive_agents: typing.Iterable[str]) -> None:
        """Specify which agents are alive

        This is used to determine which callbacks to call.

        Parameters
        ----------
        alive_agents : typing.Iterable[str]
            Agents that are currently alive.
        """
        self._agent_filter = alive_agents

    def _reduce(self, r, **kwargs):
        self._reduce_fn = self._reduce_fn or np.any
        tmp = StateDict.stack_values(r)
        tmp = {k: self._reduce_fn(v, **kwargs) for k, v in tmp.items()}
        return StateDict(sorted(tmp.items()))


class InfoDict(EnvDict):
    """[summary]

    Parameters
    ----------
    EnvDict : [type]
        [description]
    """

    def _reduce(self, r, **kwargs):
        self._reduce_fn = self._reduce_fn or (lambda x: {})
        tmp = StateDict.stack_values(r)
        tmp = {k: self._reduce_fn(v, **kwargs) for k, v in tmp.items()}
        return StateDict(sorted(tmp.items()), recurse=False)
