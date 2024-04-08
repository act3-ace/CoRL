"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import corl.simulators.six_dof.base_six_dof_properties as six_dof_props


def test_6dof_default_props():
    six_dof_props.LatLonProp()
    six_dof_props.LatLonProp()
    six_dof_props.AltitudePropMeters(high=[1000.0])
    six_dof_props.AltitudeRateProp(low=[-1000.0], high=[1000.0], unit="meter / second")
    six_dof_props.OrientationProp()
    six_dof_props.OrientationRateProp()
    six_dof_props.FuelProp()
    six_dof_props.MachProp(high=[2.0])
    six_dof_props.KcasProp(high=[2000.0])
    six_dof_props.TrueAirSpeedProp(high=[3000.0])
    six_dof_props.VelocityNEDProp()
    six_dof_props.AccelerationNEDProp()
    six_dof_props.FlightPathAngleProp()
    six_dof_props.FlightPathAngleRadProp()
    six_dof_props.GLoadProp()
    six_dof_props.GloadNzProp()
    six_dof_props.GloadNyProp()
    six_dof_props.AngleOfAttackProp()
    six_dof_props.AngleOfAttackRateProp()
    six_dof_props.AngleOfSideSlipProp()
    six_dof_props.AngleOfSideSlipRateProp()
    six_dof_props.FuelWeightProp()
    six_dof_props.WindDirectionProp()
    six_dof_props.WindSpeedProp()
