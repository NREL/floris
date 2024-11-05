import numpy as np
import pytest

from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT
from floris.core.turbine.tum_operation_model import TUMLossTurbine
from floris.utilities import cosd
from tests.conftest import SampleInputs, WIND_SPEEDS


def test_submodel_attributes():

    assert hasattr(TUMLossTurbine, "power")
    assert hasattr(TUMLossTurbine, "thrust_coefficient")
    assert hasattr(TUMLossTurbine, "axial_induction")

def test_TUMLossTurbine():

    # NOTE: These tests should be updated to reflect actual expected behavior
    # of the TUMLossTurbine model. Currently, match the CosineLossTurbine model.

    n_turbines = 1
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine
    turbine_data["power_thrust_table"] = SampleInputs().tum_loss_turbine_power_thrust_table

    yaw_angles_nom = 0 * np.ones((1, n_turbines))
    tilt_angles_nom = turbine_data["power_thrust_table"]["ref_tilt"] * np.ones((1, n_turbines))
    power_setpoints_nom = POWER_SETPOINT_DEFAULT * np.ones((1, n_turbines))
    yaw_angles_test = 20 * np.ones((1, n_turbines))
    tilt_angles_test = 0 * np.ones((1, n_turbines))


    # Check that power works as expected
    TUMLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"], # Matches ref_air_density
        yaw_angles=yaw_angles_nom,
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    # truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    # baseline_power = turbine_data["power_thrust_table"]["power"][truth_index] * 1000
    # assert np.allclose(baseline_power, test_power)

    # Check that yaw and tilt angle have an effect
    TUMLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"], # Matches ref_air_density
        yaw_angles=yaw_angles_test,
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_test,
        tilt_interp=None
    )
    #assert test_power < baseline_power

    # Check that a lower air density decreases power appropriately
    TUMLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1,
        yaw_angles=yaw_angles_nom,
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    #assert test_power < baseline_power


    # Check that thrust coefficient works as expected
    TUMLossTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
        yaw_angles=yaw_angles_nom,
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    #baseline_Ct = turbine_data["power_thrust_table"]["thrust_coefficient"][truth_index]
    #assert np.allclose(baseline_Ct, test_Ct)

    # Check that yaw and tilt angle have the expected effect
    TUMLossTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
        yaw_angles=yaw_angles_test,
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_test,
        tilt_interp=None
    )
    #absolute_tilt = tilt_angles_test - turbine_data["power_thrust_table"]["ref_tilt"]
    #assert test_Ct == baseline_Ct * cosd(yaw_angles_test) * cosd(absolute_tilt)


    # Check that thrust coefficient works as expected
    TUMLossTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
        yaw_angles=yaw_angles_nom,
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    (
        cosd(yaw_angles_nom)
        * cosd(tilt_angles_nom - turbine_data["power_thrust_table"]["ref_tilt"])
    )
    # baseline_ai = (
    #     1 - np.sqrt(1 - turbine_data["power_thrust_table"]["thrust_coefficient"][truth_index])
    # ) / 2 / baseline_misalignment_loss
    # assert np.allclose(baseline_ai, test_ai)

    # Check that yaw and tilt angle have the expected effect
    TUMLossTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
        yaw_angles=yaw_angles_test,
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_test,
        tilt_interp=None
    )
    tilt_angles_test - turbine_data["power_thrust_table"]["ref_tilt"]
    #assert test_Ct == baseline_Ct * cosd(yaw_angles_test) * cosd(absolute_tilt)
