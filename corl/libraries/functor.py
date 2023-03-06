"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import enum
from collections.abc import Mapping as Mapping_Type
from string import digits
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, Field, PyObject, validator
from typing_extensions import Annotated

from corl.libraries.factory import Factory
from corl.libraries.parameters import Parameter
from corl.libraries.units import NoneUnitType, ValueWithUnits
from corl.simulators.base_platform import BasePlatform

ObjectStoreElem = Union[Parameter, ValueWithUnits, Any]


class Functor(BaseModel):
    """
    - name: The optional name of the functor.
    - class: The class of the functor.
    - config: The functor's specific configuration dictionary.
    """
    functor: PyObject
    name: Annotated[str, Field(strip_whitespace=True, min_length=1)] = ''
    config: Dict[str, ObjectStoreElem] = {}
    references: Dict[str, str] = {}

    # This might not be necessary after Parameter has pydantic validation
    class Config:
        """Allow arbitrary types for Parameter"""
        arbitrary_types_allowed = True
        extra = 'forbid'

    @validator('name', pre=True, always=True)
    def name_from_functor(cls, v, values):
        """Create default value for name from the functor name"""
        # "'functor' not in values" means PyObject failed and an error is coming
        # Check for it here because that error is more helpful than a KeyError here.
        if not v and 'functor' in values:
            return values['functor'].__name__
        return v

    resolve_factory = validator('config', pre=True, each_item=True, allow_reuse=True)(Factory.resolve_factory)

    def create_functor_object(
        self, param_sources: Sequence[Mapping[str, Any]] = (), ref_sources: Sequence[Mapping[str, Any]] = (), **kwargs
    ):
        """ Create the object with this functor

        TODO:  Better description
        """
        functor_args = self.resolve_storage_and_references(
            param_sources=param_sources,
            ref_sources=ref_sources,
            platform=kwargs.get("platform", None),
        )
        return self.functor(name=self.name, **functor_args, **kwargs)

    def resolve_storage_and_references(
        self,
        param_sources: Sequence[Mapping[str, Any]] = (),
        ref_sources: Sequence[Mapping[str, Any]] = (),
        platform: BasePlatform = None,
    ):
        """ Resolve parameter storage and references to get direct functor arguments."""

        functor_args: Dict[str, Any] = {}
        functor_units = getattr(self.functor, 'REQUIRED_UNITS', {})

        # Is there a units field in the configuration file?
        config_units = self.config.get('unit', None)
        if config_units is not None and not isinstance(config_units, str):
            raise TypeError(f'Units of {self.name} are "{config_units}", which is not a string')

        for arg_name, arg_value in self.config.items():

            # Resolve parameter values
            if isinstance(arg_value, Parameter):
                for source in param_sources:
                    if arg_name in source.get(self.name, {}):
                        resolved_value = source[self.name][arg_name]
                        break
                else:
                    # This "else" means "no break encountered", which means that the argument was not found in any source
                    raise RuntimeError(f'Could not resolve argument {arg_name} for {self.name} from the parameter sources')
            else:
                resolved_value = arg_value

            # Resolve units
            functor_args[arg_name] = self._resolve_units(
                name=arg_name, value=resolved_value, functor_units=functor_units, config_units=config_units, error_name=self.name
            )

        if platform:
            remove_digits = str.maketrans('', '', digits)
            team_string = platform.name.translate(remove_digits)

        for ref_dest, ref_src in self.references.items():
            true_dest = ref_dest.split(".")
            true_ref_src = ref_src
            if platform:
                true_ref_src = true_ref_src.replace("%%SIDE%%", team_string).replace("%%PLATFORM%%", platform.name)

            # Resolve references
            for source in ref_sources:
                if true_ref_src in source:
                    ref_obj = source[true_ref_src]
                    if not isinstance(ref_obj, Parameter):
                        break
            else:
                # This "else" means "no break encountered", which means either:
                # 1. true_ref_src was not found in any source
                # 2. true_ref_src was found; however, it was an unresolved Parameter
                raise RuntimeError(f'Could not find {ref_src} -> {true_ref_src} for {self.name} in the reference storage')

            # Resolve units
            tmp_functor_args = functor_args
            for functor_arg_path in true_dest[:-1]:
                tmp_functor_args = tmp_functor_args.setdefault(functor_arg_path, {})
            tmp_functor_args[true_dest[-1]] = self._resolve_units(
                name=ref_dest, value=ref_obj, functor_units=functor_units, config_units=config_units, error_name=self.name
            )

        return functor_args

    def add_to_parameter_store(self, parameter_store: Dict[str, Dict[str, Parameter]]) -> None:
        """Add the parameters of this functor to an external parameter store.

        Parameters
        ----------
        parameter_store : Dict[str, Dict[str, Parameter]]
            Parameter store to which to add the parameters.  The keys of the outer dictionary are functor names.  The inner dictionary is
            the collection of Parameters.
        """
        parameters = {k: v for k, v in self.config.items() if isinstance(v, Parameter)}
        if parameters:
            if self.name in parameter_store:
                raise ValueError(f'Duplicate functor name in parameter store: {self.name}')
            parameter_store[self.name] = parameters

    @staticmethod
    def _resolve_units(
        name: str,
        value: ObjectStoreElem,
        functor_units: Dict[str, Union[None, Literal[True], enum.Enum]],
        config_units: Optional[str],
        error_name: str
    ) -> Any:

        assert not isinstance(value, Parameter)

        functor_arg: Any

        # Determine what to pass in to functor
        if isinstance(value, ValueWithUnits):
            # Value has units
            if name in functor_units:
                # Extra local variable required so that MyPy does proper type reduction in the conditional block below.
                these_units = functor_units[name]
                if these_units is True:
                    # Require it to be actually boolean True, not just "truthy"
                    functor_arg = value
                else:
                    functor_arg = value.as_units(these_units)
            elif config_units is not None:
                functor_arg = value.as_units(config_units)
            elif value.units is NoneUnitType.NoneUnit:
                functor_arg = value.value
            else:
                raise RuntimeError(f'Argument {name} of {error_name} has units {value.units}, but none are expected')
        else:
            # Value has no units
            # Functor requires them and the unit is not None
            if functor_units.get(name, NoneUnitType.NoneUnit) not in [None, NoneUnitType.NoneUnit]:
                raise RuntimeError(f'Argument {name} of {error_name} is missing required units')

            # Functor does not require them
            functor_arg = value

        return functor_arg


class FunctorWrapper(Functor):
    """
    wrapped: The functor or functor wrapper configuration wrapped by this functor wrapper.
    """
    wrapped: Union['FunctorMultiWrapper', 'FunctorWrapper', 'FunctorDictWrapper', Functor]

    def create_functor_object(
        self, param_sources: Sequence[Mapping[str, Any]] = (), ref_sources: Sequence[Mapping[str, Any]] = (), **kwargs
    ):
        """ Create the object with this functor

        TODO:  Better description
        TODO:  Clean up logic so does not have too-many-branches
        """

        wrapped_func = self.wrapped.create_functor_object(param_sources=param_sources, ref_sources=ref_sources, **kwargs)

        functor_args = self.resolve_storage_and_references(
            param_sources=param_sources,
            ref_sources=ref_sources,
            platform=kwargs.get("platform", None),
        )

        return self.functor(name=self.name, wrapped=wrapped_func, **functor_args, **kwargs)

    def add_to_parameter_store(self, parameter_store: Dict[str, Dict[str, Parameter]]) -> None:
        """Add the parameters of this functor to an external parameter store.

        Parameters
        ----------
        parameter_store : Dict[str, Dict[str, Parameter]]
            Parameter store to which to add the parameters.  The keys of the outer dictionary are functor names.  The inner dictionary is
            the collection of Parameters.
        """
        super().add_to_parameter_store(parameter_store)
        self.wrapped.add_to_parameter_store(parameter_store)


class FunctorMultiWrapper(Functor):
    """
    wrapped: The functor or functor wrapper configuration wrapped by this functor wrapper.
    """
    wrapped: List[Union['FunctorMultiWrapper', FunctorWrapper, Functor, 'FunctorDictWrapper']]

    def create_functor_object(
        self, param_sources: Sequence[Mapping[str, Any]] = (), ref_sources: Sequence[Mapping[str, Any]] = (), **kwargs
    ):
        """ Create the object with this functor

        TODO:  Better description
        TODO:  Clean up logic so does not have too-many-branches
        """
        wrapped_funcs = [x.create_functor_object(param_sources=param_sources, ref_sources=ref_sources, **kwargs) for x in self.wrapped]

        functor_args = self.resolve_storage_and_references(
            param_sources=param_sources,
            ref_sources=ref_sources,
            platform=kwargs.get("platform", None),
        )

        return self.functor(name=self.name, wrapped=wrapped_funcs, **functor_args, **kwargs)

    def add_to_parameter_store(self, parameter_store: Dict[str, Dict[str, Parameter]]) -> None:
        """Add the parameters of this functor to an external parameter store.

        Parameters
        ----------
        parameter_store : Dict[str, Dict[str, Parameter]]
            Parameter store to which to add the parameters.  The keys of the outer dictionary are functor names.  The inner dictionary is
            the collection of Parameters.
        """
        super().add_to_parameter_store(parameter_store)
        for wrapped in self.wrapped:
            wrapped.add_to_parameter_store(parameter_store)


class FunctorDictWrapper(Functor):
    """
    wrapped: The dict of functor or functor wrapper configurations wrapped by this wrapper
    """
    wrapped: Dict[str, Union['FunctorDictWrapper', FunctorMultiWrapper, FunctorWrapper, Functor]]

    @validator('wrapped', pre=True)
    def adjust_wrapped_name(cls, v):
        """Use the dictionary wrapping key as the name if none is provided."""
        if not isinstance(v, Mapping_Type):
            return v
        # this pre validator is pretty dangerous in terms of
        # what it will attempt to run on, because a regular wrapper
        # has a dictionary it will attempt to perform this operation
        # on regular wrappers, so we are just going to verify that
        # this is not a regular wrapper by checking for a functor appearing
        # in wrapped
        if "functor" in v.keys():
            return v
        for key, value in v.items():
            try:
                if 'name' not in value:
                    value['name'] = key
            except:  # pylint: disable=bare-except  # noqa: E722
                pass
        return v

    def create_functor_object(
        self, param_sources: Sequence[Mapping[str, Any]] = (), ref_sources: Sequence[Mapping[str, Any]] = (), **kwargs
    ):
        """ Create the object with this functor

        TODO:  Better description
        TODO:  Clean up logic so does not have too-many-branches
        """
        wrapped_funcs = {
            k: v.create_functor_object(param_sources=param_sources, ref_sources=ref_sources, **kwargs)
            for k, v in self.wrapped.items()
        }

        functor_args = self.resolve_storage_and_references(
            param_sources=param_sources,
            ref_sources=ref_sources,
            platform=kwargs.get("platform", None),
        )

        return self.functor(name=self.name, wrapped=wrapped_funcs, **functor_args, **kwargs)

    def add_to_parameter_store(self, parameter_store: Dict[str, Dict[str, Parameter]]) -> None:
        """Add the parameters of this functor to an external parameter store.

        Parameters
        ----------
        parameter_store : Dict[str, Dict[str, Parameter]]
            Parameter store to which to add the parameters.  The keys of the outer dictionary are functor names.  The inner dictionary is
            the collection of Parameters.
        """
        super().add_to_parameter_store(parameter_store)
        for wrapped in self.wrapped.values():
            wrapped.add_to_parameter_store(parameter_store)


FunctorWrapper.update_forward_refs()
FunctorMultiWrapper.update_forward_refs()
FunctorDictWrapper.update_forward_refs()
