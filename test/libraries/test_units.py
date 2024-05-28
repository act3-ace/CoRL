# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
from typing import Annotated

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from corl.libraries.units import Quantity, corl_get_ureg, corl_quantity, is_compatible, set_req_unit


@pytest.mark.parametrize(
    "u1_val, u1_unit, u2_val, u2_unit, truth",
    [
        pytest.param(5, "degree", 5, "meter", False),
        pytest.param(5, "degree", 5, "dimensionless", False),
        pytest.param(2, "radian", 2, "dimensionless", False),
        pytest.param(1, "radian", 1, "radian", True),
        pytest.param(1, "radian", 1, "degree", True),
        pytest.param(1, "g0", 2, "standard_gravity", True),
    ],
)
def test_unit_compatibility_quantity(u1_val, u1_unit, u2_val, u2_unit, truth):
    q1 = corl_quantity()(u1_val, u1_unit)
    q2 = corl_quantity()(u2_val, u2_unit)
    assert q1.is_compatible_with(q2.units) == truth
    assert is_compatible(q1, q2) == truth
    assert is_compatible(q1, u2_unit) == truth
    assert is_compatible(q2, u1_unit) == truth


@pytest.mark.parametrize(
    "value,unit,base_unit,passes_validation,validated_value",
    [
        pytest.param(5, "degree", "meter", False, None),
        pytest.param(5, "degree", "dimensionless", False, None),
        pytest.param(2, "radian", "dimensionless", False, None),
        pytest.param(1, "radian", "radian", True, 1),
        pytest.param(1, "radian", "degree", True, 57.29577951),
        pytest.param(1, "g0", "standard_gravity", True, 1),
        pytest.param(1, "g0", None, True, 1),
    ],
)
def test_quantity_validator(value, unit, base_unit, passes_validation, validated_value):
    class TestValidator(BaseModel):
        test_value: Annotated[Quantity, set_req_unit(base_unit)]

    try:
        validated = TestValidator(test_value={"value": value, "unit": unit, "dtype": "float32"})
        assert np.isclose(validated.test_value.m, validated_value)
        expected_units = base_unit if base_unit is not None else unit
        assert corl_get_ureg().get_unit(validated.test_value.units) == corl_get_ureg().get_unit(expected_units)
        assert passes_validation is True
    except ValidationError:
        assert passes_validation is False
    except KeyError:
        assert passes_validation is False
