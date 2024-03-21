"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------

Structures that hold parameters and the ability to update them.
"""
import abc
import typing
import warnings
from typing import runtime_checkable

import numpy as np
from numpy.random import Generator, RandomState
from pydantic import BaseModel, ConfigDict, PositiveFloat, StrictInt, TypeAdapter, field_validator, model_validator
from scipy import stats
from typing_extensions import Protocol

from corl.libraries.factory import Factory
from corl.libraries.units import Quantity, corl_get_ureg

Number = StrictInt | float
Randomness = Generator | RandomState
OtherVars = typing.Mapping[tuple[str, ...], Quantity]


@runtime_checkable
class _ConstraintCallbackType(Protocol):
    def __call__(self, old_arg: Number, new_arg: Number) -> Number:
        ...


class ParameterValidator(BaseModel):
    """Validator class for Parameter"""

    units: str = "dimensionless"
    update: dict[str, typing.Any] = {}
    simulator: dict[str, typing.Any] = {}
    episode_parameter_provider: dict[str, typing.Any] = {}
    dependent_parameters: dict[str, tuple[str, ...]] = {}

    @field_validator("dependent_parameters", mode="before")
    @classmethod
    def dependent_parameters_check(cls, v):
        """Validate the units field"""
        v2 = TypeAdapter(dict[str, str | typing.Sequence[str]]).validate_python(v)
        return {key: ("reference_store", item) if isinstance(item, str) else tuple(item) for key, item in v2.items()}


class Parameter(abc.ABC):
    """Parameter class"""

    def __init__(self, **kwargs) -> None:
        self.config: ParameterValidator = self.get_validator()(**kwargs)

        # Create and save updaters
        self.updaters: dict[str, typing.Any] = {}
        for name, val in self.config.update.items():
            factory = Factory(**val)
            self.updaters[name] = factory.build(param=self, name=name, constraint=self.get_constraint(name=name))

    @staticmethod
    def get_validator() -> type[ParameterValidator]:
        """Get the validator class for this Parameter"""
        return ParameterValidator

    def get_constraint(self, name: str) -> _ConstraintCallbackType | None:  # noqa: PLR6301
        """Get the constraint function for this Parameter's updater config"""
        return None

    @abc.abstractmethod
    def get_value(self, rng: Randomness, other_vars: OtherVars) -> str | Quantity:
        """Get the value of the parameter.

        In order to avoid inconsistent operation between cases where the value is serialized (such as from the ray object store) or not,
        this method should not modify the attributes of `self`.  This is the equivalent of a C++ `const` method for developers familiar with
        that concept.

        Parameters
        ----------
        rng : Union[Generator, RandomState]
            Random number generator from which to draw random values.
        other_vars: variables previously processed, will contain any variables this parameter is dependent on
        """

    # @staticmethod
    # def _serialize(v) -> str:
    #     return str(v)

    # @classmethod
    # def __get_pydantic_core_schema__(cls, source, handler):
    #     serializer = core_schema.plain_serializer_function_ser_schema(cls._serialize, when_used='json')
    #     if cls is source:
    #         # Treat bare usage of ImportString (`schema is None`) as the same as ImportString[Any]
    #         return core_schema.no_info_plain_validator_function(
    #             function=cls.validate, serialization=serializer
    #         )
    #     else:
    #         return core_schema.no_info_before_validator_function(
    #             function=cls.validate, schema=handler(source), serialization=serializer
    #         )

    # @classmethod
    # def validate(cls, v):
    #     if isinstance(v, Parameter):
    #         return v
    #     if not isinstance(v, dict):
    #         # msg = "Quantity validator input must either be a Quantity or Dict"
    #         # raise TypeError(msg)
    #         return v

    #     return tmp


class ParameterWrapperValidator(BaseModel):
    """Validator class for Parameter"""

    wrapped: dict[str, Parameter]
    # simulator: typing.Dict[str, typing.Any] = {}
    # episode_parameter_provider: typing.Dict[str, typing.Any] = {}
    dependent_parameters: dict[str, tuple[str, ...]] = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ParameterWrapper(abc.ABC):
    """ParameterWrapper class"""

    def __init__(self, **kwargs) -> None:
        self.config: ParameterWrapperValidator = self.get_validator()(**kwargs)

    @staticmethod
    def get_validator() -> type[ParameterWrapperValidator]:
        """Get the validator class for this Parameter"""
        return ParameterWrapperValidator

    def get_constraint(self, name: str) -> _ConstraintCallbackType | None:  # noqa: PLR6301
        """Get the constraint function for this Parameter's updater config"""
        return None

    @abc.abstractmethod
    def get_value(self, rng: Randomness, other_vars: OtherVars) -> dict[str, str | Quantity]:
        """Get the value of the parameters this Wrapper holds.

        This return object should contain a mapping to each parameter this wrapper wraps

        *IMPORTANT NOTE ON USE
        The environment will truncate the parameter tuple off the end and insert the string of the
        parameter key dict

        self.config.wrapped = {
            lat: ConstantParam(foo)
            lon: ConstantParam(bar)
        }

        ie: (group1, latlon (name of parameter)) -> (group1, lat) and (group2, lon)
        # END NOTE

        Parameters
        ----------
        rng : Union[Generator, RandomState]
            Random number generator from which to draw random values.
        other_vars: variables previously processed, will contain any variables this parameter is dependent on
        """


class PassthroughParameterWrapper(ParameterWrapper):
    """
    This parameter wrapper simply takes outputs the parameters it wraps
    into a dict
    """

    def get_value(self, rng, other_vars):
        return {k: v.get_value(rng, other_vars) for k, v in self.config.wrapped.items()}


class ConstantParameterValidator(ParameterValidator):
    """Validator class for ConstantParameter"""

    value: Number | str


class ConstantParameter(Parameter):
    """A parameter that always has a constant value."""

    def __init__(self, **kwargs) -> None:
        self.config: ConstantParameterValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[ConstantParameterValidator]:
        return ConstantParameterValidator

    def get_value(self, rng: Randomness, other_vars: OtherVars):
        if isinstance(self.config.value, str):
            return self.config.value
        return corl_get_ureg().Quantity(value=self.config.value, units=self.config.units)


class UniformParameterValidator(ParameterValidator):
    """Validator class for UniformParameter"""

    low: Number
    high: Number

    @model_validator(mode="after")
    def high_validator(self):
        """Validate the high field"""
        assert self.high > self.low, "Upper bound must not be smaller than lower bound"
        return self


class UniformParameter(Parameter):
    """A parameter that draws from a uniform distribution."""

    def __init__(self, **kwargs) -> None:
        self.config: UniformParameterValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[UniformParameterValidator]:
        return UniformParameterValidator

    def get_value(self, rng: Randomness, other_vars: OtherVars):
        return corl_get_ureg().Quantity(value=rng.uniform(self.config.low, self.config.high), units=self.config.units)

    def get_constraint(self, name: str) -> _ConstraintCallbackType | None:
        if name == "low":
            return self._min_with_high
        if name == "high":
            return self._max_with_low
        raise ValueError("Unknown constraint name")

    def _min_with_high(self, old_arg: Number, new_arg: Number) -> Number:
        if new_arg > self.config.high:
            warnings.warn("Could not fully update UniformParameter lower bound as it exceeds higher bound")
            return self.config.high
        return new_arg

    def _max_with_low(self, old_arg: Number, new_arg: Number) -> Number:
        if new_arg < self.config.low:
            warnings.warn("Could not fully update UniformParameter lower bound as it goes below the lower bound")
            return self.config.low
        return new_arg


class StepUniformParameterValidator(UniformParameterValidator):
    """Validator class for UniformParameter"""

    step: Number


class StepUniformParameter(Parameter):
    """A parameter that draws from a uniform distribution plus step."""

    def __init__(self, **kwargs) -> None:
        self.config: UniformParameterValidator
        super().__init__(**kwargs)
        if self.config.low == self.config.high:
            self._choices = [self.config.low]
        else:
            self._choices = list(range(self.config.low, self.config.high, self.config.step))  # type: ignore
        if self.config.high not in self._choices:
            self._choices.append(self.config.high)

    @staticmethod
    def get_validator() -> type[StepUniformParameterValidator]:
        return StepUniformParameterValidator

    def get_value(self, rng: Randomness, other_vars: OtherVars):
        return corl_get_ureg().Quantity(value=rng.choice(self._choices), units=self.config.units)


class TruncatedNormalParameterValidator(ParameterValidator):
    """Validator class for TruncatedNormalParameter"""

    mu: Number
    std: PositiveFloat
    half_width_factor: PositiveFloat


class TruncatedNormalParameter(Parameter):
    """A parameter that draws from a truncated normal distribution."""

    def __init__(self, **kwargs) -> None:
        self.config: TruncatedNormalParameterValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[TruncatedNormalParameterValidator]:
        return TruncatedNormalParameterValidator

    def get_value(self, rng: Randomness, other_vars: OtherVars) -> Quantity:
        low = self.config.mu - self.config.half_width_factor * self.config.std
        high = self.config.mu + self.config.half_width_factor * self.config.std
        value = stats.truncnorm.rvs(
            (low - self.config.mu) / self.config.std,
            (high - self.config.mu) / self.config.std,
            loc=self.config.mu,
            scale=self.config.std,
            size=1,
            random_state=rng,
        )[0]
        return corl_get_ureg().Quantity(value=value, units=self.config.units)

    def get_constraint(self, name: str) -> _ConstraintCallbackType | None:
        if name == "std":
            return self._std_positive
        if name == "half_width_factor":
            return self._half_width_factor_positive
        raise ValueError("Unknown constraint name")

    @staticmethod
    def _generic_positive(variable: str, old_arg: Number, new_arg: Number) -> Number:
        if new_arg < 0:
            warnings.warn(f"Could not update TruncatedNormalParameter {variable} because it is not strictly positive")
            return old_arg
        return new_arg

    def _std_positive(self, old_arg: Number, new_arg: Number) -> Number:
        return self._generic_positive(variable="standard deviation", old_arg=old_arg, new_arg=new_arg)

    def _half_width_factor_positive(self, old_arg: Number, new_arg: Number) -> Number:
        return self._generic_positive(variable="half width factor", old_arg=old_arg, new_arg=new_arg)


class ChoiceParameterValidator(ParameterValidator):
    """Validator for ChoiceParameter"""

    choices: typing.Sequence[typing.Any]


class ChoiceParameter(Parameter):
    """A parameter drawn uniformly from a collection of discrete values.
    This parameter does not support updaters.

    Parameters
    ----------
    hparams : dict
        The hyperparameters that define this parameter.  In addition to the structure specified by the base class Parameter, there needs
        to be the following fields, expressed as YAML:

        ```yaml
        choices: Sequence[Any]
        ```
    """

    def __init__(self, **kwargs) -> None:
        self.config: ChoiceParameterValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator() -> type[ChoiceParameterValidator]:
        return ChoiceParameterValidator

    def get_value(self, rng: Randomness, other_vars: OtherVars):
        val = rng.choice(self.config.choices)
        if isinstance(val, str):
            return val
        return corl_get_ureg().Quantity(value=val, units=self.config.units)


class OverridableParameterWrapper(Parameter):
    """A Parameter that wraps another parameter and can override its output."""

    def __init__(self, base: Parameter | ParameterWrapper) -> None:
        # Base class API
        if isinstance(base, ParameterWrapper):
            raise RuntimeError(
                "OverridableParameterWrapper Does not support ParameterWrapper"
                "Evaluation test cases should be fully defined and should probably not use"
                "parameter wrappers"
            )
        self.config = base.config
        self.updaters = base.updaters

        # Other attributes
        self.base = base
        self.override_value: typing.Any = None

    @staticmethod
    def get_validator() -> typing.NoReturn:
        """OverridableParameterWrapper is not parsed using validators."""
        raise NotImplementedError()

    def get_constraint(self, name: str) -> _ConstraintCallbackType | None:
        return self.base.get_constraint(name=name)

    def get_value(self, rng: Randomness, other_vars: OtherVars):
        base_value = self.base.get_value(rng, other_vars)

        if self.override_value is not None:
            if isinstance(self.override_value, str):
                return self.override_value
            assert isinstance(base_value, Quantity)
            if isinstance(self.override_value, Quantity):
                return self.override_value.to(base_value.u)
            return corl_get_ureg().Quantity(value=self.override_value, units=base_value.u)

        return base_value


class UpdaterValidator(BaseModel):
    """Validator class for Updater"""

    name: str
    param: Parameter
    constraint: _ConstraintCallbackType = lambda old_arg, new_arg: new_arg
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("param")
    def param_hasattr(cls, v, info):
        """Validator for param field"""
        assert hasattr(v.config, info.data["name"]), f'Attribute {info.data["name"]} does not exist'
        return v

    @field_validator("constraint", mode="before")
    @classmethod
    def constraint_not_none(cls, v):
        """Conversion for constraint of None to identity"""
        return (lambda old_arg, new_arg: new_arg) if v is None else v


class Updater(abc.ABC):
    """Generic structure to define the method of updating a hyperparameter of a `Parameter`."""

    def __init__(self, **kwargs) -> None:
        # Validate and save updater configuration data
        self.config: UpdaterValidator = self.get_validator()(**kwargs)

    def __call__(self, *, reverse: bool = False) -> None:
        """Perform an update of the hyperparameter according to the functionality of the updater.

        The generic functionality of an update has four steps:
          1. Get the current value of the hyperparameter by extracting the named attribute from the connected parameter.
          2. Perform the update according to the `do_call` method within the updater.  The current value of the hyperparameter is provided,
             along with the flag to specify reverse updates.
          3. Apply the constraint function.  The primary purpose of this function is to allow the `Parameter` to provide a function to
             ensure that the updated hyperparameter value leaves the `Parameter` in a consistent state.  It receives the old argument and
             the new argument as keyword arguments (with names `old_arg` and `new_arg` respectively).  Common uses are to ensure that
             multiple hyperparameters within an `Parameter` maintain a consistent state (i.e., lower bound is less than upper bound) or that
             some invariant property is maintained (i.e., standard deviation is positive).  The constraint function must return the value
             that should be set into the hyperparameter.
          4. The output of the constraint function is set back to the named attribute in the connected parameter.

        Subclasses should not override `__call__`, but should implement `do_call`.

        Parameters
        ----------
        reverse : bool, optional
            Perform a reverse update, by default False
        """
        old_arg = self.get_current_extent()
        new_arg = self.do_call(old_arg, reverse=reverse)
        constrained_arg = self.config.constraint(old_arg=old_arg, new_arg=new_arg)
        setattr(self.config.param.config, self.config.name, constrained_arg)

    def update_to_bound(self) -> None:
        """Update all the way to the bound."""
        old_arg = self.get_current_extent()
        new_arg = self.get_bound()
        constrained_arg = self.config.constraint(old_arg=old_arg, new_arg=new_arg)
        setattr(self.config.param.config, self.config.name, constrained_arg)

    @abc.abstractmethod
    def do_call(self, arg: Number, *, reverse: bool = False) -> Number:
        """Perform the update of the provided hyperparameter.

        Parameters
        ----------
        arg : Union[int, float]
            The current value of the hyperparameter.
        reverse: bool, optional
            Perform a reverse update.  If allowed, performing a sequence with a normal update followed by a reverse update should return
            the original value.  Not all `Updater` subclasses allow reverse updates.  The default is False.
        """

    @staticmethod
    def get_validator() -> type[UpdaterValidator]:
        """Get the validator for this class"""
        return UpdaterValidator

    @abc.abstractmethod
    def supports_reverse_update(self) -> bool:
        """Indicator whether this updater supports reverse updates."""

    @abc.abstractmethod
    def at_bound(self) -> bool:
        """Indicator whether this updater is at its bound."""

    @abc.abstractmethod
    def get_bound(self) -> Number:
        """Get the bound of this updater."""

    def get_current_extent(self) -> Number:
        """returns the current extent of the parameter controlled by this updater

        Returns:
            Number -- The current extent of the parameter controlled by this updater
        """
        return getattr(self.config.param.config, self.config.name)

    @abc.abstractmethod
    def create_config(self) -> dict:
        """Create the configuration file that would generate this object in its current state.

        Returns
        -------
        dict
            Configuration file of the current state of the object.  This object can be passed to the
            constructor of the Parameter to regenerate it.
        """
        raise NotImplementedError()


class BoundStepUpdaterValidator(UpdaterValidator):
    """Validator class for BoundStepUpdater"""

    bound_type: typing.Literal["min", "max"]
    bound: Number
    step: Number

    @model_validator(mode="after")
    def step_validator(self):
        """Validator for step field"""
        if self.step >= 0 and self.bound_type == "min":
            raise ValueError("Step must be negative for minimum bound")
        if self.step <= 0 and self.bound_type == "max":
            raise ValueError("Step must be positive for maximum bound")
        return self


class BoundStepUpdater(Updater):
    """An `Updater` that advances by a constant step, limited by a bound.

    On each update, the provided value is incremented by a step size.  If that increment causes it to violate the provided bound, the value
    is given the value of that bound.

    Reverse updates are supported.  They are bounded at the initial value of the provided parameter.
    """

    def __init__(self, **kwargs) -> None:
        self.config: BoundStepUpdaterValidator
        super().__init__(**kwargs)

        self._reverse_bound = self.get_current_extent()

        self._bound_func: typing.Callable
        self._reverse_bound_func: typing.Callable
        if self.config.bound_type == "min":
            self._bound_func = max
            self._reverse_bound_func = min
        elif self.config.bound_type == "max":
            self._bound_func = min
            self._reverse_bound_func = max
        else:
            raise ValueError(f'Unknown bound type {self.config["bound_type"]}')
        self._at_bound: bool = False

    @staticmethod
    def get_validator() -> type[BoundStepUpdaterValidator]:
        """Get validator for BoundStepUpdater"""
        return BoundStepUpdaterValidator

    def do_call(self, arg: Number, *, reverse: bool = False) -> Number:
        if reverse:
            arg -= self.config.step
            output = self._reverse_bound_func(arg, self._reverse_bound)
        else:
            arg += self.config.step
            output = self._bound_func(arg, self.config.bound)

        self._at_bound = np.isclose(output, self.config.bound)

        return output

    def supports_reverse_update(self) -> bool:  # noqa: PLR6301
        return True

    def update_to_bound(self) -> None:
        super().update_to_bound()
        self._at_bound = True

    def at_bound(self) -> bool:
        return self._at_bound

    def get_bound(self) -> Number:
        return self.config.bound

    def create_config(self) -> dict:
        return {"bound": self.config.bound, "step": self.config.step, "bound_type": self.config.bound_type}


class RandomSignUniformParameter(UniformParameter):
    """A parameter that draws from a uniform distribution where the value is assigned a random sign.
    The range of the parameter is [-high, -low], [low, high]."""

    def get_value(self, rng: Randomness, other_vars: OtherVars):
        return corl_get_ureg().Quantity(value=rng.uniform(self.config.low, self.config.high) * rng.choice([-1, 1]), units=self.config.units)
