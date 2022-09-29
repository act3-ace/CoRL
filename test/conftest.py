"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
# https://docs.pytest.org/en/2.7.3/plugins.html?highlight=re

import os
import tempfile
from contextlib import contextmanager
from typing import List

import pytest
import ray

from corl.episode_parameter_providers import EpisodeParameterProvider
from corl.episode_parameter_providers.remote import RemoteEpisodeParameterProvider


@pytest.fixture(scope="session", autouse=True)
def ray_session(request):
    # first things first, it turns out the num_gpus override does not currently
    # work in the ray.init() method:
    #
    #   https://github.com/ray-project/ray/issues/8460
    #
    # I experienced this first-hand. I figured out a workaround using putenv,
    # which we should probably do in unit tests that use ray.init(). In this
    # test it likely would have zero effect, since no training or evaluation
    # is called, but it's a good practice to be in.
    os.putenv("CUDA_VISIBLE_DEVICES", "-1")
    global _ray_session_temp_dir
    with tempfile.TemporaryDirectory() as _ray_session_temp_dir:
        ray_config = {"address": None, "include_dashboard": False, "num_gpus": 0, "_temp_dir": _ray_session_temp_dir, "_redis_password": None, "ignore_reinit_error": False}
        ray.init(**ray_config)
        yield
        ray.shutdown()


@pytest.fixture
def self_managed_ray():
    """Enable a test to manage their own ray initialization.

    The `ray_session` fixture above ensures that all tests have a properly initialized ray
    environment.  However, some tests need more control over the ray configuration for the duration
    of the test.  The most common example is tests that need to specify `local_mode=True` within
    the evaluation.  The trivial implementation of these tests is to put `ray.shutdown` at the
    beginning of the test and then to configure ray for that particular test.  The problem with this
    approach is that it does not restore ray to a properly initialized state for any other unit test
    that assumes that the `ray_session` fixture had properly initialized ray.

    Therefore, the recommended approach for any test that needs to manage their own ray
    configuration is to use this fixture.  It automatically ensures that ray is not active at the
    beginning of the test and ensures that ray is restored to the expected configuration afterwards.
    """

    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()
    ray_config = {"include_dashboard": False, "num_gpus": 0, "_temp_dir": _ray_session_temp_dir}
    ray.init(**ray_config)


@pytest.fixture
def optional_debuggable_ray(request):
    """Fixture to easily allow debugging of unit tests that run ray.

    If @pytest.mark.ray_debug_mode(True), this is the same as if self_managed_ray was used except that it yields True.
    Otherwise, it yields False and does nothing.
    """
    marker = request.node.get_closest_marker("ray_debug_mode")
    if marker is None:
        raise RuntimeError('Use of fixture "optional_debuggable_ray" requires "@pytest.mark.ray_debug_mode(True/False)"')
    value = marker.args[0]
    if value:
        if ray.is_initialized():
            ray.shutdown()
        yield True
        if ray.is_initialized():
            ray.shutdown()
        ray_config = {"include_dashboard": False, "num_gpus": 0, "_temp_dir": _ray_session_temp_dir}
        ray.init(**ray_config)
    else:
        yield False
