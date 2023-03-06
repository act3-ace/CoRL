"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
EpisodeParameterProvider that reads data from a tabular structure.
"""
import copy
import typing

import pandas as pd
from pydantic import validator

from corl.episode_parameter_providers import EpisodeParameterProvider, EpisodeParameterProviderValidator, ParameterModel, Randomness
from corl.libraries.parameters import OverridableParameterWrapper


class TabularValidator(EpisodeParameterProviderValidator):
    """Validation model for the inputs of TabularParameterProvider"""
    separator: str = '/'
    filename: typing.Optional[str] = None
    read_csv_kwargs: typing.Dict[str, typing.Any] = {}
    data: pd.DataFrame = None

    @validator('separator')
    def separator_validator(cls, v):
        """Validate that the separator is a single character"""
        if len(v) != 1:
            raise ValueError('Separator must be single character')
        return v

    @validator('data', pre=True, always=True)
    def data_validator(cls, v, values):
        """Validate the data field"""
        if v is None:
            if values['filename'] is None:
                raise ValueError('Either data or filename must be provided')
            v = pd.read_csv(values['filename'], **values['read_csv_kwargs'])
        return v

    @validator('data')
    def column_validator(cls, v, values):
        """Validate the columns in the data field"""
        parameters_keys = {values['separator'].join(x) for x in values['parameters'].keys()}
        data_columns = set(v.columns)
        if parameters_keys != data_columns:
            left_side = parameters_keys - data_columns
            right_side = data_columns - parameters_keys
            raise RuntimeError(
                "TabularParameterProvider data table columns do not match the parameters keys, "
                f"parameters currently expected are {parameters_keys}\n"
                f"parameters not covered by config are {left_side}\n"
                f"parameters that are extra in config are {right_side}"
            )
        # assert parameters_keys == data_columns, "TabularParameterProvider data table columns do not match the parameters keys"
        return v


class TabularParameterProvider(EpisodeParameterProvider):
    """EpisodeParameterProvider that reads data from a tabular structure."""

    def __init__(self, **kwargs) -> None:
        self.config: TabularValidator
        super().__init__(**kwargs)

        self._params = {k: OverridableParameterWrapper(v) for k, v in self.config.parameters.items()}

        self._index = 0
        self._episode_id = 0

    @property
    def get_validator(self) -> typing.Type[TabularValidator]:
        """Get the validator for this class."""
        return TabularValidator

    def _do_get_params(self, rng: Randomness) -> typing.Tuple[ParameterModel, typing.Union[int, None]]:

        self._episode_id = self.config.data.index[self._index] if self._index < len(self.config.data) else None

        row = self.config.data.iloc[self._index % len(self.config.data)]
        self._index += 1

        pm = copy.deepcopy(self._params)
        for i in row.index:
            pm[tuple(i.split(self.config.separator))].override_value = row.loc[i]

        return pm, self._episode_id
