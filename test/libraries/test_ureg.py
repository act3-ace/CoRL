# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
import numpy as np
import pytest

from corl.libraries.units import UnitRegistryConfiguration


def test_basic_ureg():
    ureg = UnitRegistryConfiguration().create_registry_from_args()

    _Quantity = ureg.Quantity

    val1 = _Quantity(5.0, "feet")
    val2 = val1.to("meter")
    assert val2.m == pytest.approx(1.524)
    val3 = val2.to("feet")
    assert val3.m == pytest.approx(val1.m)

    val1 = _Quantity(5.0, "meter")
    val2 = val1.to("feet")
    assert val2.m == pytest.approx(16.4042)
    val3 = val2.to("meter")
    assert val3.m == pytest.approx(val1.m)

    new_val = np.array([1.0, 2.0, 3.0])
    val1 = _Quantity(new_val, "g0")
    val2 = val1.to("m/s**2")
    assert val1.m * 9.80665 == pytest.approx(val2.m)
    val3 = val2.to("g0")
    assert val1.m == pytest.approx(val3.m)

    new_val = np.array([1, 2, 3], np.float32)
    val1 = _Quantity(new_val, "meter")
    val2 = val1.to("nmi")
    assert val2.m == pytest.approx(val1.m / 1852.0018)
    val3 = val2.to("meter")
    assert val3.m == pytest.approx(val1.m)

    new_val = np.array([1, 2, 3], np.float32)
    val1 = _Quantity(new_val, "foot")
    val2 = val1.to("nmi")
    assert val2.m == pytest.approx(val1.m / 6076.1216)
    val3 = val2.to("feet")
    assert val3.m == pytest.approx(val1.m)

    new_val = np.array([1, 2, 3], np.float32)
    val1 = _Quantity(new_val, "radian")
    val2 = val1.to("degree")
    assert val2.m == pytest.approx(val1.m * 57.2958)
    val3 = val2.to("radian")
    assert val3.m == pytest.approx(val1.m)


def test_ureg_wraps():
    ureg = UnitRegistryConfiguration().create_registry_from_args()

    _Quantity = ureg.Quantity

    def toy_function(in1, string_in2, in3, string_in4):
        return in1 * 2, string_in2[0], in3 // 2, string_in4[-1]

    new_func = ureg.wraps(("meter", None, "feet", None), ("meter", None, "feet", None), toy_function)

    in1 = _Quantity(6.56168, "feet")
    in2 = "hello"
    in3 = _Quantity(2, "meter")
    in4 = "world"

    out1, out2, out3, out4 = new_func(in1, in2, in3, in4)
    assert str(out1.u) == "meter"
    assert out1.m == pytest.approx(4)
    assert out2 == "h"
    assert str(out3.u) == "feet"
    assert out3.m == pytest.approx(3.0)
    assert out4 == "d"


def test_ureg_wraps_with_default():
    ureg = UnitRegistryConfiguration().create_registry_from_args()

    _Quantity = ureg.Quantity

    def toy_function(in1, string_in2, in3, string_in4, bool_in5=True):
        return in1 * 2, string_in2[0], in3 // 2, string_in4[-1]

    new_func = ureg.wraps(("meter", None, "feet", None), ("meter", None, "feet", None, None), toy_function)

    in1 = _Quantity(6.56168, "feet")
    in2 = "hello"
    in3 = _Quantity(2, "meter")
    in4 = "world"

    out1, out2, out3, out4 = new_func(in1, in2, in3, in4)
    assert str(out1.u) == "meter"
    assert out1.m == pytest.approx(4)
    assert out2 == "h"
    assert str(out3.u) == "feet"
    assert out3.m == pytest.approx(3.0)
    assert out4 == "d"


def test_ureg_wraps_with_default_but_no_kwarg():
    ureg = UnitRegistryConfiguration().create_registry_from_args()

    _Quantity = ureg.Quantity

    def toy_function(in1, string_in2, in3, string_in4, bool_in5=True):
        return in1 * 2, string_in2[0], in3 // 2, string_in4[-1]

    new_func = ureg.wraps(("meter", None, "feet", None), ("meter", None, "feet", None, None), toy_function)

    in1 = _Quantity(6.56168, "feet")
    in2 = "hello"
    in3 = _Quantity(2, "meter")
    in4 = "world"

    out1, out2, out3, out4 = new_func(in1, in2, in3, in4, True)
    assert str(out1.u) == "meter"
    assert out1.m == pytest.approx(4)
    assert out2 == "h"
    assert str(out3.u) == "feet"
    assert out3.m == pytest.approx(3.0)
    assert out4 == "d"


def test_ureg_wraps_with_kwargs():
    ureg = UnitRegistryConfiguration().create_registry_from_args()

    _Quantity = ureg.Quantity

    def toy_function(in1, string_in2, in3, string_in4, bool_in5=True):
        return in1 * 2, string_in2[0], in3 // 2, string_in4[-1]

    new_func = ureg.wraps(("meter", None, "feet", None), ("meter", None, "feet", None, None), toy_function)

    in1 = _Quantity(6.56168, "feet")
    in2 = "hello"
    in3 = _Quantity(2, "meter")
    in4 = "world"

    out1, out2, out3, out4 = new_func(in1=in1, string_in2=in2, in3=in3, string_in4=in4, bool_in5=True)
    assert str(out1.u) == "meter"
    assert out1.m == pytest.approx(4)
    assert out2 == "h"
    assert str(out3.u) == "feet"
    assert out3.m == pytest.approx(3.0)
    assert out4 == "d"
