"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3)/Autonomous Air Combat Operations
Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------
"""

from corl.libraries import units
import pytest


@pytest.mark.parametrize(
    'initial_value, initial_units, resolved_units',
    [
        pytest.param(float(1.1), 'feet', units.Distance.Feet),
        pytest.param(int(1), units.Speed.Knots, units.Speed.Knots),
        pytest.param(True, units.Angle.Degree, units.Angle.Degree),
        pytest.param('test_string', None, units.NoneUnitType.NoneUnit)
    ]
)
def test_storage_types(initial_value, initial_units, resolved_units):
    val = units.ValueWithUnits(value=initial_value, units=initial_units)
    assert type(val.value) == type(initial_value)
    assert val.value == initial_value
    assert val.units == resolved_units


units_to_test = []
for dim in units.Dimensions:
    for unit in dim:
        for str_unit in unit.value[1]:
            units_to_test.append(pytest.param(str_unit, unit, id=f'{unit.name}_{str_unit}'))

@pytest.mark.parametrize('test_str,truth', units_to_test)
def test_storage_strings(test_str, truth):
    test_unit = units.GetUnitFromStr(test_str)
    assert test_unit == truth


# These should be the "most usual" conversion so that anyone reading this file can confirm that the
# test is proper.
@pytest.mark.parametrize(
    'initial_value, initial_units, converted_value, converted_units',
    [
        pytest.param(1.0, units.Distance.Feet, 12 * 2.54 / 100, units.Distance.Meter, id='feet_to_meter'),
        pytest.param(1.0, units.Distance.Nautical_Mile, 1852, units.Distance.Meter, id='nautical_mile_to_meter'),
        pytest.param(180.0, units.Angle.Degree, 3.14159, units.Angle.Rad, id='degree_to_radian'),
        pytest.param(1800.0, units.Time.Second, 0.5, units.Time.Hour, id='second_to_hour'),
        pytest.param(1.0, units.Speed.Meter_per_Sec, 1.94384, units.Speed.Knots, id='m/s_to_knots'),
        pytest.param(12 * 2.54, units.Speed.Meter_per_Sec, 6000, units.Speed.Feet_per_Min, id='m/s_to_ft/min'),
        pytest.param(12 * 2.54, units.Speed.Meter_per_Sec, 100, units.Speed.Feet_per_Sec, id='m/s_to_ft/s'),
        pytest.param(90.0, units.AngularSpeed.degrees_per_sec, 3.14159 / 2, units.AngularSpeed.radians_per_sec, id='deg/s_to_rad/s'),
        pytest.param(1.94384, units.Acceleration.knots_per_sec, 1, units.Acceleration.meter_per_sec_2, id='knot/s_to_m/s2'),
        pytest.param(9.80665, units.Acceleration.meter_per_sec_2, 1.0, units.Acceleration.standard_gravity, id='m/s2_to_g'),
        pytest.param(1.0, units.Weight.Kilogram, 2.20462, units.Weight.Pound, id='kg_to_lb'),
    ]
)
def test_conversion(initial_value, initial_units, converted_value, converted_units):
    val = units.ValueWithUnits(value=initial_value, units=initial_units)
    assert val.as_units(converted_units) == pytest.approx(converted_value)


@pytest.mark.parametrize(
    'initial_value, initial_units, case, principal_value',
    [
        pytest.param(-270, units.Angle.Degree, units.ValueWithUnits.PrincipalValueNormalization.Positive, 90),
        pytest.param(-135, units.Angle.Degree, units.ValueWithUnits.PrincipalValueNormalization.Positive, 225),
        pytest.param(135, units.Angle.Degree, units.ValueWithUnits.PrincipalValueNormalization.Positive, 135),
        pytest.param(270, units.Angle.Degree, units.ValueWithUnits.PrincipalValueNormalization.Positive, 270),
        pytest.param(405, units.Angle.Degree, units.ValueWithUnits.PrincipalValueNormalization.Positive, 45),
        pytest.param(-270, units.Angle.Degree, units.ValueWithUnits.PrincipalValueNormalization.Centered, 90),
        pytest.param(-135, units.Angle.Degree, units.ValueWithUnits.PrincipalValueNormalization.Centered, -135),
        pytest.param(135, units.Angle.Degree, units.ValueWithUnits.PrincipalValueNormalization.Centered, 135),
        pytest.param(270, units.Angle.Degree, units.ValueWithUnits.PrincipalValueNormalization.Centered, -90),
        pytest.param(405, units.Angle.Degree, units.ValueWithUnits.PrincipalValueNormalization.Centered, 45),
    ]
)
def test_normalize(initial_value, initial_units, case, principal_value):
    val = units.ValueWithUnits(value=initial_value, units=initial_units)
    val.normalize_to_principal_value(case)
    assert val.value == pytest.approx(principal_value)
