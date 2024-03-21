import os
import tempfile
from pathlib import Path

import pydantic
import pytest

from corl.libraries.file_path import CorlFilePath


class CorlFilePathTestModel(pydantic.BaseModel):
    test_path: CorlFilePath


@pytest.fixture()
def file_resource():
    tmp_file = tempfile.NamedTemporaryFile()
    yield tmp_file


def test_various_file_paths(file_resource):
    # verify the most basic case works
    file_resouce_path = Path(file_resource.name)
    tmp = CorlFilePathTestModel(test_path=str(file_resouce_path))
    assert tmp.test_path == file_resouce_path
    # verify we can resolve env variables
    file_res_name = file_resouce_path.name
    file_res_dir = file_resouce_path.parents[0]
    print(file_res_dir, file_res_name)

    os.environ["CORL_TEST_DIR"] = str(file_res_dir)

    test_string = f"${{CORL_TEST_DIR}}/{file_res_name}"
    tmp = CorlFilePathTestModel(test_path=test_string)
    assert tmp.test_path == file_resouce_path

    test_string = f"${{CORL_TEST_DIR}}/{file_res_name}"
    tmp = CorlFilePathTestModel(test_path=test_string)
    assert tmp.test_path == file_resouce_path


def test_cwd_file_paths():
    test_file = Path(Path.cwd(), "config", "tasks", "pong", "rllib_config.yml")

    assert test_file.exists(), "this unit test relies on the existence of a file, make sure test_file exists"
    test_string = "config/tasks/pong/rllib_config.yml"
    tmp = CorlFilePathTestModel(test_path=test_string)
    assert tmp.test_path == test_file

    test_string = "./config/tasks/pong/rllib_config.yml"
    tmp = CorlFilePathTestModel(test_path=test_string)
    assert tmp.test_path == test_file

    test_string = "./config/tasks/../tasks/pong/../pong/rllib_config.yml"
    tmp = CorlFilePathTestModel(test_path=test_string)
    assert tmp.test_path == test_file

    os.environ["CORL_TEST_LOC"] = "tasks/pong"

    test_string = "./config/${CORL_TEST_LOC}/rllib_config.yml"
    tmp = CorlFilePathTestModel(test_path=test_string)
    assert tmp.test_path == test_file
