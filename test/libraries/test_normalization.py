"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import numpy as np
import pytest
from gym import spaces

from corl.libraries.normalization import Normalizer, LinearNormalizer, StandardNormalNormalizer

@pytest.mark.parametrize("norm_class, config, raw_space, norm_space",
    [
        pytest.param(LinearNormalizer,
            {
                "minimum": -2.0,
                "maximum": 2.0
            },
            spaces.Box(low=np.array([-1.0, -5.0, -10.0]), high=np.array([1.0, 5.0, 7.0]), dtype=np.float32),
            spaces.Box(low=np.array([-2.0, -2.0, -2.0]), high=np.array([2.0, 2.0, 2.0]), dtype=np.float32),
            id="LinearNormalizer"),
        pytest.param(StandardNormalNormalizer,
            {
                "mu": [0.0, -1.0, -5.0],
                "sigma": [1.0, 1.0, 10.0],
            },
            spaces.Box(low=np.array([-1.0, -5.0, -10.0]), high=np.array([1.0, 5.0, 7.0]), dtype=np.float32),
            spaces.Box(low=np.array([-1.0, -4.0, -0.5]), high=np.array([1.0, 6.0, 1.2]), dtype=np.float32),
            id="StandardNormalNormalizer",
            ),
    ]
)
def test_normalization(norm_class: Normalizer, config: dict, raw_space: spaces.Box, norm_space: spaces.Box):
    normalizer: Normalizer = norm_class(**config)

    raw_sample = raw_space.sample()

    assert norm_space == normalizer.normalize_space(raw_space)

    norm_sample = normalizer.normalize_sample(raw_space, raw_sample)

    assert np.all(np.isclose(raw_sample, normalizer.unnormalize_sample(raw_space, norm_sample), atol=.0001))
