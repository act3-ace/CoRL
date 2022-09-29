"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Structures that hold parameters and the ability to update them.
"""
import abc
import enum
import typing
import warnings

import numpy as np
from numpy.random import Generator, RandomState
from pydantic import BaseModel, PositiveFloat, StrictInt, validator
from scipy import stats
from typing_extensions import Protocol, runtime_checkable

from corl.libraries import units
from corl.libraries.factory import Factory
from corl.libraries.units import ValueWithUnits

Number = typing.Union[StrictInt, float]
Randomness = typing.Union[Generator, RandomState]


@runtime_checkable
class _ConstraintCallbackType(Protocol):

    def __call__(self, old_arg: Number, new_arg: Number) -> Number:
        ...


class ParameterValidator(BaseModel):
    """Validator class for Parameter"""
    units: typing.Optional[enum.Enum]
    update: typing.Dict[str, typing.Any] = {}
    simulator: typing.Dict[str, typing.Any] = {}
    episode_parameter_provider: typing.Dict[str, typing.Any] = {}

    @validator('units', pre=True)
    def units_validator(cls, v):
        """Validate the units field"""
        return units.GetUnitFromStr(v) if v is not None else None


class Parameter(abc.ABC):
    """Parameter class"""

    def __init__(self, **kwargs) -> None:
        self.config: ParameterValidator = self.get_validator(**kwargs)

        # Create and save updaters
        self.updaters: typing.Dict[str, typing.Any] = {}
        for name, val in self.config.update.items():
            factory = Factory(**val)
            self.updaters[name] = factory.build(param=self, name=name, constraint=self.get_constraint(name=name))

    @property
    def get_validator(self) -> typing.Type[ParameterValidator]:
        """Get the validator class for this Parameter"""
        return ParameterValidator

    def get_constraint(self, name: str) -> typing.Optional[_ConstraintCallbackType]:  # pylint: disable=unused-argument
        """Get the constraint function for this Parameter's updater config"""
        return None

    @abc.abstractmethod
    def get_value(self, rng: Randomness) -> units.ValueWithUnits:
        """Get the value of the parameter.

        In order to avoid inconsistent operation between cases where the value is serialized (such as from the ray object store) or not,
        this method should not modify the attributes of `self`.  This is the equivalent of a C++ `const` method for developers familiar with
        that concept.

        Parameters
        ----------
        rng : Union[Generator, RandomState]
            Random number generator from which to draw random values.
        """
        ...


class ConstantParameterValidator(ParameterValidator):
    """Validator class for ConstantParameter"""
    value: typing.Union[Number, str]


class ConstantParameter(Parameter):
    """A parameter that always has a constant value."""

    def __init__(self, **kwargs) -> None:
        self.config: ConstantParameterValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[ConstantParameterValidator]:
        return ConstantParameterValidator

    def get_value(self, rng: Randomness) -> units.ValueWithUnits:
        return units.ValueWithUnits(value=self.config.value, units=self.config.units)


class UniformParameterValidator(ParameterValidator):
    """Validator class for UniformParameter"""
    low: Number
    high: Number

    @validator("high")
    def high_validator(cls, v, values):
        """Validate the high field"""
        if v < values['low']:
            raise ValueError('Upper bound must not be smaller than lower bound')
        return v


class UniformParameter(Parameter):
    """A parameter that draws from a uniform distribution."""

    def __init__(self, **kwargs) -> None:
        self.config: UniformParameterValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[UniformParameterValidator]:
        return UniformParameterValidator

    def get_value(self, rng: Randomness) -> units.ValueWithUnits:
        return units.ValueWithUnits(value=rng.uniform(self.config.low, self.config.high), units=self.config.units)

    def get_constraint(self, name: str) -> typing.Optional[_ConstraintCallbackType]:
        if name == 'low':
            return self._min_with_high
        if name == 'high':
            return self._max_with_low
        raise ValueError("Unknown contraint name")

    def _min_with_high(self, old_arg: Number, new_arg: Number) -> Number:  # pylint: disable=unused-argument
        if new_arg > self.config.high:
            warnings.warn('Could not fully update UniformParameter lower bound as it exceeds higher bound')
            return self.config.high
        return new_arg

    def _max_with_low(self, old_arg: Number, new_arg: Number) -> Number:  # pylint: disable=unused-argument
        if new_arg < self.config.low:
            warnings.warn('Could not fully update UniformParameter lower bound as it goes below the lower bound')
            return self.config.low
        return new_arg


class TruncatedNormalParameterValidator(ParameterValidator):
    """Validator class for TruncatedNormalParameter"""
    mu: Number
    std: PositiveFloat
    half_width_factor: PositiveFloat


class TruncatedNormalParameter(Parameter):
    """A parameter that draws from a truncated normal distribution.
    """

    def __init__(self, **kwargs) -> None:
        self.config: TruncatedNormalParameterValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[TruncatedNormalParameterValidator]:
        return TruncatedNormalParameterValidator

    def get_value(self, rng: Randomness) -> units.ValueWithUnits:
        low = self.config.mu - self.config.half_width_factor * self.config.std
        high = self.config.mu + self.config.half_width_factor * self.config.std
        value = stats.truncnorm.rvs(
            (low - self.config.mu) / self.config.std, (high - self.config.mu) / self.config.std,
            loc=self.config.mu,
            scale=self.config.std,
            size=1,
            random_state=rng
        )[0]
        return units.ValueWithUnits(value=value, units=self.config.units)

    def get_constraint(self, name: str) -> typing.Optional[_ConstraintCallbackType]:
        if name == 'std':
            return self._std_positive
        if name == 'half_width_factor':
            return self._half_width_factor_positive
        raise ValueError("Unknown contraint name")

    @staticmethod
    def _generic_positive(variable: str, old_arg: Number, new_arg: Number) -> Number:
        if new_arg < 0:
            warnings.warn(f'Could not update TruncatedNormalParameter {variable} because it is not strictly positive')
            return old_arg
        return new_arg

    def _std_positive(self, old_arg: Number, new_arg: Number) -> Number:
        return self._generic_positive(variable='standard deviation', old_arg=old_arg, new_arg=new_arg)

    def _half_width_factor_positive(self, old_arg: Number, new_arg: Number) -> Number:
        return self._generic_positive(variable='half width factor', old_arg=old_arg, new_arg=new_arg)


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

    @property
    def get_validator(self) -> typing.Type[ChoiceParameterValidator]:
        return ChoiceParameterValidator

    def get_value(self, rng: Randomness) -> units.ValueWithUnits:
        return units.ValueWithUnits(value=rng.choice(self.config.choices), units=self.config.units)


class OverridableParameterWrapper(Parameter):
    """A Parameter that wraps another parameter and can override its output."""

    def __init__(self, base: Parameter) -> None:  # pylint: disable=super-init-not-called
        # Base class API
        self.config = base.config
        self.updaters = base.updaters

        # Other attributes
        self.base = base
        self.override_value: typing.Any = None

    @property
    def get_validator(self) -> typing.NoReturn:
        """OverridableParameterWrapper is not parsed using validators."""
        raise NotImplementedError()

    def get_constraint(self, name: str) -> typing.Optional[_ConstraintCallbackType]:
        return self.base.get_constraint(name=name)

    def get_value(self, rng: Randomness) -> units.ValueWithUnits:

        if self.override_value is not None:
            return ValueWithUnits(value=self.override_value, units=self.config.units)

        return self.base.get_value(rng)


class UpdaterValidator(BaseModel):
    """Validator class for Updater"""
    name: str
    param: Parameter
    constraint: _ConstraintCallbackType = lambda old_arg, new_arg: new_arg

    class Config:
        """pydantic Config class"""
        arbitrary_types_allowed = True

    @validator('param')
    def param_hasattr(cls, v, values):
        """Validator for param field"""
        assert hasattr(v.config, values['name']), f'Attribute {values["name"]} does not exist'
        return v

    @validator('constraint', pre=True)
    def constraint_not_none(cls, v):
        """Conversion for constraint of None to identity"""
        if v is None:
            return lambda old_arg, new_arg: new_arg
        return v


class Updater(abc.ABC):
    """Generic structure to define the method of updating a hyperparameter of a `Parameter`."""

    def __init__(self, **kwargs) -> None:
        # Validate and save updater configuration data
        self.config: UpdaterValidator = self.get_validator(**kwargs)

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
        ...

    @property
    def get_validator(self) -> typing.Type[UpdaterValidator]:
        """Get the validator for this class"""
        return UpdaterValidator

    @abc.abstractmethod
    def supports_reverse_update(self) -> bool:
        """Indicator whether this updater supports reverse updates."""
        ...

    @abc.abstractmethod
    def at_bound(self) -> bool:
        """Indicator whether this updater is at its bound."""
        ...

    @abc.abstractmethod
    def get_bound(self) -> Number:
        """Get the bound of this updater."""
        ...

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

    @validator('step')
    def step_validator(cls, v, values):
        """Validator for step field"""
        if v >= 0 and values['bound_type'] == 'min':
            raise ValueError('Step must be negative for minimum bound')
        if v <= 0 and values['bound_type'] == 'max':
            raise ValueError('Step must be positive for maximum bound')
        return v


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
        if self.config.bound_type == 'min':
            self._bound_func = max
            self._reverse_bound_func = min
        elif self.config.bound_type == 'max':
            self._bound_func = min
            self._reverse_bound_func = max
        else:
            raise ValueError(f'Unknown bound type {self.config["bound_type"]}')
        self._at_bound: bool = False

    @property
    def get_validator(self) -> typing.Type[BoundStepUpdaterValidator]:
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

    def supports_reverse_update(self) -> bool:
        return True

    def update_to_bound(self) -> None:
        super().update_to_bound()
        self._at_bound = True

    def at_bound(self) -> bool:
        return self._at_bound

    def get_bound(self) -> Number:
        return self.config.bound

    def create_config(self) -> dict:
        return {'bound': self.config.bound, 'step': self.config.step, 'bound_type': self.config.bound_type}
