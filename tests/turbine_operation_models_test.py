import numpy as np

from floris.simulation.turbine.operation_models import (
    CosineLossTurbine,
    rotor_velocity_air_density_correction,
    SimpleTurbine,
)
from floris.utilities import cosd
from tests.conftest import SampleInputs, WIND_SPEEDS


def test_rotor_velocity_air_density_correction():

    wind_speed = 10.
    ref_air_density = 1.225
    test_density = 1.2

    test_speed = rotor_velocity_air_density_correction(wind_speed, ref_air_density, ref_air_density)
    assert test_speed == wind_speed

    test_speed = rotor_velocity_air_density_correction(wind_speed, test_density, test_density)
    assert test_speed == wind_speed

    test_speed = rotor_velocity_air_density_correction(0., test_density, ref_air_density)
    assert test_speed == 0.

    test_speed = rotor_velocity_air_density_correction(wind_speed, test_density, ref_air_density)
    assert np.allclose((test_speed/wind_speed)**3, test_density/ref_air_density)

def test_submodel_attributes():

    assert hasattr(SimpleTurbine, "power")
    assert hasattr(SimpleTurbine, "thrust_coefficient")

    assert hasattr(CosineLossTurbine, "power")
    assert hasattr(CosineLossTurbine, "thrust_coefficient")

def test_SimpleTurbine():

    n_turbines = 1
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine

    # Check that power works as expected
    test_power = SimpleTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"], # Matches ref_air_density
    )
    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    baseline_power = turbine_data["power_thrust_table"]["power"][truth_index] * 1000
    assert np.allclose(baseline_power, test_power)

    # Check that yaw and tilt angle have no effect
    test_power = SimpleTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"], # Matches ref_air_density
        yaw_angles=20 * np.ones((1, n_turbines)),
        tilt_angles=5 * np.ones((1, n_turbines))
    )
    assert np.allclose(baseline_power, test_power)

    # Check that a lower air density decreases power appropriately
    test_power = SimpleTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1,
    )
    assert test_power < baseline_power


    # Check that thrust coefficient works as expected
    test_Ct = SimpleTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
    )
    baseline_Ct = turbine_data["power_thrust_table"]["thrust_coefficient"][truth_index]
    assert np.allclose(baseline_Ct, test_Ct)

    # Check that yaw and tilt angle have no effect
    test_Ct = SimpleTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
        yaw_angles=20 * np.ones((1, n_turbines)),
        tilt_angles=5 * np.ones((1, n_turbines))
    )
    assert np.allclose(baseline_Ct, test_Ct)


    # Check that axial induction works as expected
    test_ai = SimpleTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
    )
    baseline_ai = (
        1 - np.sqrt(1 - turbine_data["power_thrust_table"]["thrust_coefficient"][truth_index])
    )/2
    assert np.allclose(baseline_ai, test_ai)

    # Check that yaw and tilt angle have no effect
    test_ai = SimpleTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
        yaw_angles=20 * np.ones((1, n_turbines)),
        tilt_angles=5 * np.ones((1, n_turbines))
    )
    assert np.allclose(baseline_ai, test_ai)

def test_CosineLossTurbine():

    n_turbines = 1
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine

    yaw_angles_nom = 0 * np.ones((1, n_turbines))
    tilt_angles_nom = turbine_data["power_thrust_table"]["ref_tilt"] * np.ones((1, n_turbines))
    yaw_angles_test = 20 * np.ones((1, n_turbines))
    tilt_angles_test = 0 * np.ones((1, n_turbines))


    # Check that power works as expected
    test_power = CosineLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"], # Matches ref_air_density
        yaw_angles=yaw_angles_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    baseline_power = turbine_data["power_thrust_table"]["power"][truth_index] * 1000
    assert np.allclose(baseline_power, test_power)

    # Check that yaw and tilt angle have an effect
    test_power = CosineLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"], # Matches ref_air_density
        yaw_angles=yaw_angles_test,
        tilt_angles=tilt_angles_test,
        tilt_interp=None
    )
    assert test_power < baseline_power

    # Check that a lower air density decreases power appropriately
    test_power = CosineLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1,
        yaw_angles=yaw_angles_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    assert test_power < baseline_power


    # Check that thrust coefficient works as expected
    test_Ct = CosineLossTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
        yaw_angles=yaw_angles_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    baseline_Ct = turbine_data["power_thrust_table"]["thrust_coefficient"][truth_index]
    assert np.allclose(baseline_Ct, test_Ct)

    # Check that yaw and tilt angle have the expected effect
    test_Ct = CosineLossTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
        yaw_angles=yaw_angles_test,
        tilt_angles=tilt_angles_test,
        tilt_interp=None
    )
    absolute_tilt = tilt_angles_test - turbine_data["power_thrust_table"]["ref_tilt"]
    assert test_Ct == baseline_Ct * cosd(yaw_angles_test) * cosd(absolute_tilt)


    # Check that thrust coefficient works as expected
    test_ai = CosineLossTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
        yaw_angles=yaw_angles_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    baseline_misalignment_loss = (
        cosd(yaw_angles_nom)
        * cosd(tilt_angles_nom - turbine_data["power_thrust_table"]["ref_tilt"])
    )
    baseline_ai = (
        1 - np.sqrt(1 - turbine_data["power_thrust_table"]["thrust_coefficient"][truth_index])
    ) / 2 / baseline_misalignment_loss
    assert np.allclose(baseline_ai, test_ai)

    # Check that yaw and tilt angle have the expected effect
    test_ai = CosineLossTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1, # Unused
        yaw_angles=yaw_angles_test,
        tilt_angles=tilt_angles_test,
        tilt_interp=None
    )
    absolute_tilt = tilt_angles_test - turbine_data["power_thrust_table"]["ref_tilt"]
    assert test_Ct == baseline_Ct * cosd(yaw_angles_test) * cosd(absolute_tilt)
