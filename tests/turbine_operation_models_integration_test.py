import numpy as np
import pytest

from floris.core.turbine.operation_models import (
    AWCTurbine,
    CosineLossTurbine,
    MixedOperationTurbine,
    POWER_SETPOINT_DEFAULT,
    SimpleDeratingTurbine,
    SimpleTurbine,
)
from floris.utilities import cosd
from tests.conftest import SampleInputs, WIND_SPEEDS


def test_submodel_attributes():

    assert hasattr(SimpleTurbine, "power")
    assert hasattr(SimpleTurbine, "thrust_coefficient")
    assert hasattr(SimpleTurbine, "axial_induction")

    assert hasattr(CosineLossTurbine, "power")
    assert hasattr(CosineLossTurbine, "thrust_coefficient")
    assert hasattr(CosineLossTurbine, "axial_induction")

    assert hasattr(SimpleDeratingTurbine, "power")
    assert hasattr(SimpleDeratingTurbine, "thrust_coefficient")
    assert hasattr(SimpleDeratingTurbine, "axial_induction")

    assert hasattr(MixedOperationTurbine, "power")
    assert hasattr(MixedOperationTurbine, "thrust_coefficient")
    assert hasattr(MixedOperationTurbine, "axial_induction")

    assert hasattr(AWCTurbine, "power")
    assert hasattr(AWCTurbine, "thrust_coefficient")
    assert hasattr(AWCTurbine, "axial_induction")

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


def test_SimpleDeratingTurbine():

    n_turbines = 1
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine


    # Check that for no specified derating, matches SimpleTurbine
    test_Ct = SimpleDeratingTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=None,
    )
    base_Ct = SimpleTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
    )
    assert np.allclose(test_Ct, base_Ct)

    test_power = SimpleDeratingTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=None,
    )
    base_power = SimpleTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
    )
    assert np.allclose(test_power, base_power)

    test_ai = SimpleDeratingTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=None,
    )
    base_ai = SimpleTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
    )
    assert np.allclose(test_ai, base_ai)

    # When power_setpoints are 0, turbine is shut down.
    test_Ct = SimpleDeratingTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=np.zeros((1, n_turbines)),
    )
    assert np.allclose(test_Ct, 0)

    test_power = SimpleDeratingTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=np.zeros((1, n_turbines)),
    )
    assert np.allclose(test_power, 0)

    test_ai = SimpleDeratingTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=np.zeros((1, n_turbines)),
    )
    assert np.allclose(test_ai, 0)

    # When power setpoints are less than available, results should be less than when no setpoint
    wind_speed = 20 # High, so that turbine is above rated nominally
    derated_power = 4.0e6
    rated_power = 5.0e6
    test_power = SimpleDeratingTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=derated_power * np.ones((1, n_turbines)),
    )

    rated_power = 5.0e6
    test_Ct = SimpleDeratingTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=derated_power * np.ones((1, n_turbines)),
    )
    base_Ct = SimpleTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=derated_power * np.ones((1, n_turbines)),
    )
    assert np.allclose(test_Ct, derated_power/rated_power * base_Ct) # Is this correct?

    # Mixed below and above rated
    n_turbines = 2
    wind_speeds_test = np.ones((1, n_turbines, 3, 3))
    wind_speeds_test[0,0,:,:] = 20.0 # Above rated
    wind_speeds_test[0,1,:,:] = 5.0 # Well below eated
    test_power = SimpleDeratingTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speeds_test, # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=derated_power * np.ones((1, n_turbines)),
    )
    base_power = SimpleTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speeds_test, # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=derated_power * np.ones((1, n_turbines)),
    )

    assert test_power[0,0] < base_power[0,0]
    assert test_power[0,0] == derated_power

    assert test_power[0,1] == base_power[0,1]
    assert test_power[0,1] < derated_power

def test_MixedOperationTurbine():

    n_turbines = 1
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine
    tilt_angles_nom = turbine_data["power_thrust_table"]["ref_tilt"] * np.ones((1, n_turbines))

    # Check that for no specified derating or yaw angle, matches SimpleTurbine
    test_Ct = MixedOperationTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=POWER_SETPOINT_DEFAULT * np.ones((1, n_turbines)),
        yaw_angles=np.zeros((1, n_turbines)),
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    base_Ct = SimpleTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
    )

    assert np.allclose(test_Ct, base_Ct)

    test_power = MixedOperationTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=POWER_SETPOINT_DEFAULT * np.ones((1, n_turbines)),
        yaw_angles=np.zeros((1, n_turbines)),
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    base_power = SimpleTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
    )
    assert np.allclose(test_power, base_power)

    test_ai = MixedOperationTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=POWER_SETPOINT_DEFAULT * np.ones((1, n_turbines)),
        yaw_angles=np.zeros((1, n_turbines)),
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    base_ai = SimpleTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
    )
    assert np.allclose(test_ai, base_ai)

    # Check that when power_setpoints are set, matches SimpleDeratingTurbine,
    # while when yaw angles are set, matches CosineLossTurbine
    n_turbines = 2
    derated_power = 2.0e6

    test_Ct = MixedOperationTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=np.array([[POWER_SETPOINT_DEFAULT, derated_power]]),
        yaw_angles=np.array([[20.0, 0.0]]),
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    base_Ct_dr = SimpleDeratingTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=np.array([[POWER_SETPOINT_DEFAULT, derated_power]]),
    )
    base_Ct_yaw = CosineLossTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        yaw_angles=np.array([[20.0, 0.0]]),
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    base_Ct = np.array([[base_Ct_yaw[0,0], base_Ct_dr[0,1]]])
    assert np.allclose(test_Ct, base_Ct)

    # Do the same as above for power()
    test_power = MixedOperationTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=np.array([[POWER_SETPOINT_DEFAULT, derated_power]]),
        yaw_angles=np.array([[20.0, 0.0]]),
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    base_power_dr = SimpleDeratingTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=np.array([[POWER_SETPOINT_DEFAULT, derated_power]]),
    )
    base_power_yaw = CosineLossTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        yaw_angles=np.array([[20.0, 0.0]]),
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    base_power = np.array([[base_power_yaw[0,0], base_power_dr[0,1]]])
    assert np.allclose(test_power, base_power)

    # Finally, check axial induction
    test_ai = MixedOperationTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=np.array([[POWER_SETPOINT_DEFAULT, derated_power]]),
        yaw_angles=np.array([[20.0, 0.0]]),
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    base_ai_dr = SimpleDeratingTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        power_setpoints=np.array([[POWER_SETPOINT_DEFAULT, derated_power]]),
    )
    base_ai_yaw = CosineLossTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        yaw_angles=np.array([[20.0, 0.0]]),
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    base_ai = np.array([[base_ai_yaw[0,0], base_ai_dr[0,1]]])
    assert np.allclose(test_ai, base_ai)

    # Check error raised when both yaw and power setpoints are set
    with pytest.raises(ValueError):
        # Second turbine has both a power setpoint and a yaw angle
        MixedOperationTurbine.thrust_coefficient(
            power_thrust_table=turbine_data["power_thrust_table"],
            velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
            air_density=turbine_data["power_thrust_table"]["ref_air_density"],
            power_setpoints=np.array([[POWER_SETPOINT_DEFAULT, derated_power]]),
            yaw_angles=np.array([[0.0, 20.0]]),
            tilt_angles=tilt_angles_nom,
            tilt_interp=None
        )

def test_AWCTurbine():

    n_turbines = 1
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine

    # Baseline
    base_Ct = SimpleTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
    )
    base_power = SimpleTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
    )
    base_ai = SimpleTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
    )

    # Test no change to Ct, power, or ai when helix amplitudes are 0
    test_Ct = AWCTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        awc_modes=np.array([["helix"]*n_turbines]*1),
        awc_amplitudes=np.zeros((1, n_turbines)),
    )
    assert np.allclose(test_Ct, base_Ct)

    test_power = AWCTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        awc_modes=np.array([["helix"]*n_turbines]*1),
        awc_amplitudes=np.zeros((1, n_turbines)),
    )
    assert np.allclose(test_power, base_power)

    test_ai = AWCTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        awc_modes=np.array([["helix"]*n_turbines]*1),
        awc_amplitudes=np.zeros((1, n_turbines)),
    )
    assert np.allclose(test_ai, base_ai)

    # Test that Ct, power, and ai all decrease when helix amplitudes are non-zero
    test_Ct = AWCTurbine.thrust_coefficient(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        awc_modes=np.array([["helix"]*n_turbines]*1),
        awc_amplitudes=2*np.ones((1, n_turbines)),
    )
    assert test_Ct < base_Ct
    assert test_Ct > 0

    test_power = AWCTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"],
        awc_modes=np.array([["helix"]*n_turbines]*1),
        awc_amplitudes=2*np.ones((1, n_turbines)),
    )
    assert test_power < base_power
    assert test_power > 0

    test_ai = AWCTurbine.axial_induction(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        awc_modes=np.array([["helix"]*n_turbines]*1),
        awc_amplitudes=2*np.ones((1, n_turbines)),
    )
    assert test_ai < base_ai
    assert test_ai > 0
