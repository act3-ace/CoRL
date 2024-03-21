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
from pydantic import field_validator, model_validator

from corl.episode_parameter_providers import EpisodeParameterProvider, EpisodeParameterProviderValidator, ParameterModel, Randomness
from corl.libraries.parameters import OverridableParameterWrapper, Parameter


class TabularValidator(EpisodeParameterProviderValidator):
    """Validation model for the inputs of TabularParameterProvider"""

    separator: str = "/"
    filename: str | None = None
    read_csv_kwargs: dict[str, typing.Any] = {}
    data: pd.DataFrame

    @field_validator("separator")
    @classmethod
    def separator_validator(cls, v):
        """Validate that the separator is a single character"""
        if len(v) != 1:
            raise ValueError("Separator must be single character")
        return v

    @model_validator(mode="before")
    def data_validator(cls, values):
        """Validate the data field"""
        if values.get("data") is None:
            if values.get("filename") is None:
                raise ValueError("Either data or filename must be provided")
            values["data"] = pd.read_csv(values["filename"], **values["read_csv_kwargs"])
        return values

    @field_validator("data")
    def column_validator(cls, v, info):
        """Validate the columns in the data field"""
        parameters_keys = {info.data["separator"].join(x) for x in info.data["parameters"]}
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

        assert all(isinstance(v, Parameter) for v in self.config.parameters.values())
        self._params = {k: OverridableParameterWrapper(v) for k, v in self.config.parameters.items()}

        self._index: int = 0

    def reset(self) -> None:
        self._index = 0

    @staticmethod
    def get_validator() -> type[TabularValidator]:
        """Get the validator for this class."""
        return TabularValidator

    def _do_get_params(self, rng: Randomness, env_epp_ctx: dict | None) -> tuple[ParameterModel, int | None, dict | None]:
        if env_epp_ctx:
            index = env_epp_ctx["index"]
        else:
            index = self._index
            self._index += 1
            env_epp_ctx = {"index": index}

        self._episode_id = self.config.data.index[index] if index < len(self.config.data) else None

        row = self.config.data.iloc[index % len(self.config.data)]

        pm = copy.deepcopy(self._params)
        for i in row.index:
            pm[tuple(i.split(self.config.separator))].override_value = row.loc[i]

        return pm, self._episode_id, env_epp_ctx
