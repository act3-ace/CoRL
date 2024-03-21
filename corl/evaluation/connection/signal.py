"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, final
from weakref import CallableProxyType, ReferenceType, proxy, ref

T = TypeVar("T")


# class _TypeChecker:
#     def __init__(self, type_: tuple[type, ...] | Any = ()):
#         self.type_ = type_

#     def is_base_of(self, type_: "type | (_TypeChecker | Any)") -> bool:
#         if self.type_ == Any:
#             return True

#         if type_ == Any:
#             return False

#         if isinstance(type_, _TypeChecker):
#             if type_.type_ == Any:
#                 return True

#             return any(self.is_base_of(t) for t in type_.type_)

#         return issubclass(type_, self.type_)

#     def __str__(self) -> str:
#         if self.type_ == Any:
#             return f"[{Any.__qualname__}]"
#         return "[" + " | ".join(f"{t.__qualname__}" for t in self.type_) + "]"


# class _GenericType(Generic[T]):
#     @lru_cache
#     def get_generic_type(self, target: type) -> _TypeChecker:
#         for orig_base in self.__orig_bases__:
#             orig_type = get_origin(orig_base)
#             if orig_type is None:
#                 orig_type = orig_base
#             if issubclass(orig_type, target):
#                 types = get_args(orig_base)
#                 assert len(types) == 1, "There should only be one 'template' type."
#                 type_ = types[0]
#                 if type_ is None or isinstance(type_, TypeVar):
#                     return _TypeChecker(Any)

#                 arg_types = get_args(type_)
#                 if arg_types:
#                     return _TypeChecker(arg_types)

#                 return _TypeChecker((type_,))

#         return _TypeChecker()


class SlotExpired(RuntimeError):
    ...


class BaseSlot(Generic[T]):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._expired = False
        self._signals: list[ReferenceType[Signal[T]]] = []  # type: ignore

    def add_signal(self, signal: "ReferenceType[Signal[T]]"):
        if signal not in self._signals:
            self._signals.append(signal)

    def get_signals(self) -> list["ReferenceType[Signal[T]]"]:
        return self._signals

    def disconnect(self):
        self._expired = True

    def on_message(self, data: T) -> Any:
        if self._expired:
            raise SlotExpired
        return self._on_message(data)

    @abstractmethod
    def _on_message(self, data: T) -> Any:
        ...

    def __setstate__(self, state):
        self.__dict__.update(state)
        for signal_ref in self._signals:
            if (signal := signal_ref()) is not None:
                signal.register(self)


@final
class Slot(BaseSlot[T]):
    def __init__(self, callback: Callable[[T], Any], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._callback: Callable[[T], Any] = callback

    def _on_message(self, data: T) -> Any:
        return self._callback(data)


@final
class WeakSlot(BaseSlot[T]):
    def __init__(self, callback: Callable[[T], Any], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._callback: CallableProxyType[Callable[[T], Any]] = proxy(callback)

    def _on_message(self, data: T) -> Any:
        try:
            return self._callback(data)
        except ReferenceError as err:
            raise SlotExpired from err


@final
class Signal(Generic[T]):
    def __init__(self, **kwargs):
        super().__init__()
        self._slots: list[BaseSlot[T]] = []  # type: ignore

    def register(self, slot: BaseSlot[T]) -> BaseSlot[T]:
        # signal_type = self.get_generic_type(Signal)
        # slot_type = slot.get_generic_type(Slot)

        # if not slot_type.is_base_of(signal_type):
        #     raise RuntimeError(
        #         f"Mismatched types: {type(self).__name__}{signal_type} is not able to "
        #         f"register slot {type(slot).__name__}{slot_type}"
        #     )

        ## Maintain self._slots insertion order and do not allow duplicates
        if slot not in self._slots:
            self._slots.append(slot)

            slot.add_signal(ref(self))

        return slot

    def __call__(self, data: T) -> list[Any]:
        results: list[Any] = []

        to_remove: list[BaseSlot[T]] = []
        for slot in self._slots:
            try:
                results.append(slot.on_message(data))

            except SlotExpired:  # noqa: PERF203
                to_remove.append(slot)

        for slot in to_remove:
            self._slots.remove(slot)

        return results

    def __getstate__(self):
        slots = self._slots

        self._slots = []
        state = self.__dict__.copy()
        state["slots"] = slots.copy()

        self._slots = slots

        return state
