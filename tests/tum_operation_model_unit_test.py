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

def test_TUMLossTurbine_regression():
    """
    Adding a regression test so that we can work with the model and stay confident that results
    are not changing.
    """

    n_turbines = 1
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine
    turbine_data["power_thrust_table"] = SampleInputs().tum_loss_turbine_power_thrust_table

    N_test = 20
    tilt_angles_nom = turbine_data["power_thrust_table"]["ref_tilt"] * np.ones((N_test, n_turbines))
    power_setpoints_nom = POWER_SETPOINT_DEFAULT * np.ones((N_test, n_turbines))

    yaw_max = 30 # Maximum yaw to test
    yaw_angles_test = np.linspace(-yaw_max, yaw_max, N_test).reshape(-1,1)

    power_base = np.array([
        2480803.17307080,
        2604861.74554374,
        2717127.82421417,
        2816883.67907730,
        2903511.88325795,
        2976489.92724703,
        3035386.26405968,
        3079857.64410033,
        3109647.49833632,
        3124585.07963994,
        3124585.07963994,
        3109647.49833632,
        3079857.64410033,
        3035386.26405968,
        2976489.92724703,
        2903511.88325795,
        2816883.67907730,
        2717127.82421417,
        2604861.74554374,
        2480803.17307080,
    ])

    thrust_coefficient_base = np.array([
        0.64290577,
        0.65767615,
        0.67081063,
        0.68231005,
        0.69217361,
        0.70039934,
        0.70698463,
        0.71192670,
        0.71522301,
        0.71687168,
        0.71687168,
        0.71522301,
        0.71192670,
        0.70698463,
        0.70039934,
        0.69217361,
        0.68231005,
        0.67081063,
        0.65767615,
        0.64290577,
    ])

    axial_induction_base = np.array([
        0.20153901,
        0.20779287,
        0.21346846,
        0.21853124,
        0.22294734,
        0.22668459,
        0.22971369,
        0.23200938,
        0.23355155,
        0.23432623,
        0.23432623,
        0.23355155,
        0.23200938,
        0.22971369,
        0.22668459,
        0.22294734,
        0.21853124,
        0.21346846,
        0.20779287,
        0.20153901,
    ])

    power = TUMLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((N_test, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1,
        yaw_angles=yaw_angles_test,
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    ).squeeze()

    thrust_coefficient = TUMLossTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((N_test, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1,
        yaw_angles=yaw_angles_test,
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    ).squeeze()

    axial_induction = TUMLossTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((N_test, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1,
        yaw_angles=yaw_angles_test,
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    ).squeeze()

    # print(power)
    # print(thrust_coefficient)
    # print(axial_induction)

    assert np.allclose(power, power_base)
    assert np.allclose(thrust_coefficient, thrust_coefficient_base)
    assert np.allclose(axial_induction, axial_induction_base)

def test_TUMLossTurbine_integration():
    """
    Test the TUMLossTurbine model with a range of wind speeds, and then
    whether it works regardless of number of grid points.
    """

    n_turbines = 1
    turbine_data = SampleInputs().turbine
    turbine_data["power_thrust_table"] = SampleInputs().tum_loss_turbine_power_thrust_table

    N_test = 20
    tilt_angles_nom = turbine_data["power_thrust_table"]["ref_tilt"] * np.ones((N_test, n_turbines))
    power_setpoints_nom = POWER_SETPOINT_DEFAULT * np.ones((N_test, n_turbines))

    # Check runs over a range of wind speeds
    wind_speeds = np.linspace(1, 30, N_test)
    wind_speeds = np.tile(wind_speeds[:,None,None,None], (1, 1, 3, 3))

    power0 = TUMLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speeds,
        air_density=1.1,
        yaw_angles=0 * np.ones((N_test, n_turbines)),
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    ).squeeze()

    power20 = TUMLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speeds,
        air_density=1.1,
        yaw_angles=20 * np.ones((N_test, n_turbines)),
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    ).squeeze()

    assert (power0 - power20 >= -1e6).all()

    # Won't compare; just checking runs as expected
    TUMLossTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speeds,
        air_density=1.1,
        yaw_angles=0 * np.ones((N_test, n_turbines)),
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    ).squeeze()

    TUMLossTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speeds,
        air_density=1.1,
        yaw_angles=20 * np.ones((N_test, n_turbines)),
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    ).squeeze()

    # Try a set of wind speeds for 5 grid points; then 2; then a single grid point
    # without any shear
    N_test = 1
    n_turbines = 1
    tilt_angles_nom = turbine_data["power_thrust_table"]["ref_tilt"] * np.ones((N_test, n_turbines))
    power_setpoints_nom = POWER_SETPOINT_DEFAULT * np.ones((N_test, n_turbines))


    wind_speeds = 10.0 * np.ones((N_test, n_turbines, 5, 5))
    power5gp = TUMLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speeds,
        air_density=1.1,
        yaw_angles=0 * np.ones((N_test, n_turbines)),
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    ).squeeze()

    wind_speeds = 10.0 * np.ones((N_test, n_turbines, 2, 2))
    power2gp = TUMLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speeds,
        air_density=1.1,
        yaw_angles=0 * np.ones((N_test, n_turbines)),
        power_setpoints=power_setpoints_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    ).squeeze()

    assert np.allclose(power5gp, power2gp)

    # No shear information for the TUM model to use
    wind_speeds = 10.0 * np.ones((N_test, n_turbines, 1, 1))
    with pytest.raises(ValueError):
        TUMLossTurbine.power(
            power_thrust_table=turbine_data["power_thrust_table"],
            velocities=wind_speeds,
            air_density=1.1,
            yaw_angles=0 * np.ones((N_test, n_turbines)),
            power_setpoints=power_setpoints_nom,
            tilt_angles=tilt_angles_nom,
            tilt_interp=None
        )
