"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
import threading
from typing import TypeVar

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

T = TypeVar("T")


class RestConnectionValidator(BaseModel):
    port: int = 8000
    log_level: str = "info"
    delay_start: bool = False


class BaseRestConnection:
    def __init__(self, **kwargs):
        self._config: RestConnectionValidator  # type: ignore
        super().__init__(**kwargs)

        self.app = FastAPI()

        self.app.get("/")(self.read_root)

        self.app.exception_handler(RequestValidationError)(self.validation_exception_handler)

        self.config = uvicorn.Config(self.app, port=self._config.port, log_level=self._config.log_level, loop="asyncio")
        self.server = uvicorn.Server(self.config)

        self.thread = threading.Thread(target=self.server.run)

        if not self._config.delay_start:
            self.start()

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.should_exit = True
        self.thread.join()

    def validation_exception_handler(self, request: Request, exc: RequestValidationError):  # noqa: PLR6301
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )

    def read_root(self):
        return {"help": f"navigate to {self.config.host}:{self.config.port}/docs"}

    @staticmethod
    def get_validator() -> type[RestConnectionValidator]:
        return RestConnectionValidator

    # def __getstate__(self):
    #     return {}
