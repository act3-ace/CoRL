import typing
from os.path import expandvars
from pathlib import Path
from typing import Annotated

# from pydantic.validators import path_exists_validator
from pydantic import BeforeValidator, DirectoryPath, FilePath


def core_directory_path_validator(v: typing.Any):
    if isinstance(v, Path):
        v = str(v)
    if not isinstance(v, str):
        msg = "string or Path required as argument for Corl File/Directory Paths"
        raise TypeError(msg)
    vars_v = expandvars(v)
    return Path(Path.cwd(), vars_v).resolve() if vars_v.startswith(".") else Path(vars_v).resolve()


CorlFilePath = Annotated[FilePath, BeforeValidator(core_directory_path_validator)]
CorlDirectoryPath = Annotated[Path, BeforeValidator(core_directory_path_validator)]
CorlExistingDirectoryPath = Annotated[DirectoryPath, BeforeValidator(core_directory_path_validator)]
