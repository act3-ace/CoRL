"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import shutil
import site
from pathlib import Path

import pytest

from corl.parsers.yaml_loader import load_file


@pytest.fixture()
def config_test(tmpdir):
    """Fixture to execute asserts before and after a test is run"""
    # Setup:

    test_dirs = ["corl_tmp_pkg-0.0.1.dist-info", "corl_tmp_pkg"]

    for test_dir in test_dirs:
        site_pkg_path_tmp_path = Path(site.getsitepackages()[0]) / test_dir
        if site_pkg_path_tmp_path.is_dir():
            shutil.rmtree(site_pkg_path_tmp_path)
        shutil.copytree(f"test/fixture_files/parse_test/{test_dir}", site_pkg_path_tmp_path)

    yield  # this is where the testing happens

    # Teardown
    for test_dir in test_dirs:
        site_pkg_path_tmp_path = Path(site.getsitepackages()[0]) / test_dir
        if site_pkg_path_tmp_path.is_dir():
            shutil.rmtree(site_pkg_path_tmp_path)


def test_parsing(config_test):
    truth_yaml = load_file("test/fixture_files/parse_test/parse_truth.yml")
    parsed_yaml = load_file("test/fixture_files/parse_test/parse_test.yml")
    assert truth_yaml == parsed_yaml
