"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from pathlib import Path
from typing import Annotated

from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BeforeValidator, ValidationError

from corl.evaluation.connection.base_eval_connection import BaseEvalConnection, BaseEvalConnectionValidator
from corl.evaluation.connection.base_rest_connection import BaseRestConnection, RestConnectionValidator
from corl.evaluation.launchers.base_eval import EvalConfig
from corl.evaluation.runners.inference.protocol.simple_epp_update import SimpleEppUpdate
from corl.libraries.context import add_context
from corl.parsers.yaml_loader import load_file


def build_config(v, info):
    if (v is None or len(v) == 0) and (path := info.data.get("path")):
        return load_file(config_filename=path)
    return v


class _ResetSchema(BaseModel):
    path: Path | None = None
    config: Annotated[dict | None, BeforeValidator(build_config)] = None


class RestEvalConnectionValidator(RestConnectionValidator, BaseEvalConnectionValidator):
    ...


class RestEvalConnection(BaseRestConnection, BaseEvalConnection[SimpleEppUpdate]):
    def __init__(self, **kwargs):
        external_delay_start = kwargs.get("delay_start", False)
        kwargs["delay_start"] = True
        super().__init__(**kwargs)

        self.app.post("/reset/")(self.reset_config)

        self.app.post("/epp/")(self.modify_epp)

        self.app.exception_handler(ValidationError)(self.validation_error_handler)

        if not external_delay_start:
            self.start()

    @staticmethod
    def get_validator() -> type[RestEvalConnectionValidator]:
        return RestEvalConnectionValidator

    def reset_config(self, data: _ResetSchema):
        with add_context({"connection": self}):
            eval_schema = EvalConfig(path=data.path, raw_config=data.config)
            self.reset_signal(eval_schema)

        return {"message": "ok"}

    def modify_epp(self, data: SimpleEppUpdate):
        self.modify_epp_signal(data)

        return {"message": "ok"}

    def validation_error_handler(self, request: Request, exc: ValidationError):  # noqa: PLR6301
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=exc.json(),
        )
