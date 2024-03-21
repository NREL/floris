from pathlib import Path

import numpy as np
import pytest
import yaml

from floris import FlorisModel
from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_read_yaml():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    assert isinstance(fmodel, FlorisModel)

def test_assign_setpoints():

    fmodel =  FlorisModel(configuration=YAML_INPUT)
    fmodel.set(layout_x=[0, 0], layout_y=[0, 1000])

    # Test setting yaw angles via a list, integers, numpy array
    fmodel.set(yaw_angles=[[20.0, 30.0]])
    fmodel.set(yaw_angles=[[20, 30]])
    fmodel.set(yaw_angles=np.array([[20.0, 30.0]]))

    # Test setting power setpoints in various ways
    fmodel.set(power_setpoints=[[1e6, 2e6]])
    fmodel.set(power_setpoints=np.array([[1e6, 2e6]]))

    # Disable turbines
    fmodel.set(disable_turbines=[[True, False]])
    fmodel.set(disable_turbines=np.array([[True, False]]))

    # Combination
    fmodel.set(yaw_angles=[[0, 30]], power_setpoints=np.array([[1e6, None]]))

    # power_setpoints and disable_turbines (disable_turbines overrides power_setpoints)
    fmodel.set(power_setpoints=[[1e6, 2e6]], disable_turbines=[[True, False]])
    assert np.allclose(fmodel.core.farm.power_setpoints, np.array([[0.001, 2e6]]))

def test_set_run():
    """
    These tests are designed to test the set / run sequence to ensure that inputs are
    set when they should be, not set when they shouldn't be, and that the run sequence
    retains or resets information as intended.
    """

    # In FLORIS v3.2, running calculate_wake twice incorrectly set the yaw angles when the
    # first time has non-zero yaw settings but the second run had all-zero yaw settings.
    # The test below asserts that the yaw angles are correctly set in subsequent calls to run.
    fmodel = FlorisModel(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run()
    assert fmodel.core.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run()
    assert fmodel.core.farm.yaw_angles == yaw_angles

    # Verify making changes to the layout, wind speed, wind direction  and
    # turbulence intensity both before and after running the calculation
    fmodel.reset_operation()
    fmodel.set(
        layout_x=[0, 0],
        layout_y=[0, 1000],
        wind_speeds=[8, 8],
        wind_directions=[270, 270],
        turbulence_intensities=[0.06, 0.06]
    )
    assert np.array_equal(fmodel.core.farm.layout_x, np.array([0, 0]))
    assert np.array_equal(fmodel.core.farm.layout_y, np.array([0, 1000]))
    assert np.array_equal(fmodel.core.flow_field.wind_speeds, np.array([8, 8]))
    assert np.array_equal(fmodel.core.flow_field.wind_directions, np.array([270, 270]))

    # Double check that nothing has changed after running the calculation
    fmodel.run()
    assert np.array_equal(fmodel.core.farm.layout_x, np.array([0, 0]))
    assert np.array_equal(fmodel.core.farm.layout_y, np.array([0, 1000]))
    assert np.array_equal(fmodel.core.flow_field.wind_speeds, np.array([8, 8]))
    assert np.array_equal(fmodel.core.flow_field.wind_directions, np.array([270, 270]))

    # Verify that changing wind shear doesn't change the other settings above
    fmodel.set(wind_shear=0.1)
    assert fmodel.core.flow_field.wind_shear == 0.1
    assert np.array_equal(fmodel.core.farm.layout_x, np.array([0, 0]))
    assert np.array_equal(fmodel.core.farm.layout_y, np.array([0, 1000]))
    assert np.array_equal(fmodel.core.flow_field.wind_speeds, np.array([8, 8]))
    assert np.array_equal(fmodel.core.flow_field.wind_directions, np.array([270, 270]))

    # Verify that operation set-points are retained after changing other settings
    yaw_angles = 20 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(yaw_angles=yaw_angles)
    assert np.array_equal(fmodel.core.farm.yaw_angles, yaw_angles)
    fmodel.set()
    assert np.array_equal(fmodel.core.farm.yaw_angles, yaw_angles)
    fmodel.set(wind_speeds=[10, 10])
    assert np.array_equal(fmodel.core.farm.yaw_angles, yaw_angles)
    power_setpoints = 1e6 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(power_setpoints=power_setpoints)
    assert np.array_equal(fmodel.core.farm.yaw_angles, yaw_angles)
    assert np.array_equal(fmodel.core.farm.power_setpoints, power_setpoints)

    # Test that setting power setpoints through the .set() function actually sets the
    # power setpoints in the floris object
    fmodel.reset_operation()
    power_setpoints = 1e6 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(power_setpoints=power_setpoints)
    fmodel.run()
    assert np.array_equal(fmodel.core.farm.power_setpoints, power_setpoints)

    # Similar to above, any "None" set-points should be set to the default value
    power_setpoints = np.array([[1e6, None]])
    fmodel.set(layout_x=[0, 0], layout_y=[0, 1000], power_setpoints=power_setpoints)
    fmodel.run()
    assert np.array_equal(
        fmodel.core.farm.power_setpoints,
        np.array([[power_setpoints[0, 0], POWER_SETPOINT_DEFAULT]])
    )

def test_reset_operation():
    # Calling the reset function should reset the power setpoints to the default values
    fmodel = FlorisModel(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    power_setpoints = 1e6 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(power_setpoints=power_setpoints, yaw_angles=yaw_angles)
    fmodel.run()
    fmodel.reset_operation()
    assert fmodel.core.farm.yaw_angles == np.zeros(
        (fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines)
    )
    assert fmodel.core.farm.power_setpoints == (
        POWER_SETPOINT_DEFAULT * np.ones((fmodel.core.flow_field.n_findex,
                                          fmodel.core.farm.n_turbines))
    )

    # Double check that running the calculate also doesn't change the operating set points
    fmodel.run()
    assert fmodel.core.farm.yaw_angles == np.zeros(
        (fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines)
    )
    assert fmodel.core.farm.power_setpoints == (
        POWER_SETPOINT_DEFAULT * np.ones((fmodel.core.flow_field.n_findex,
                                          fmodel.core.farm.n_turbines))
    )

def test_run_no_wake():
    # In FLORIS v3.2, running calculate_no_wake twice incorrectly set the yaw angles when the first
    # time has non-zero yaw settings but the second run had all-zero yaw settings. The test below
    # asserts that the yaw angles are correctly set in subsequent calls to run_no_wake.
    fmodel = FlorisModel(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run_no_wake()
    assert fmodel.core.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run_no_wake()
    assert fmodel.core.farm.yaw_angles == yaw_angles

    # With no wake and three turbines in a line, the power for all turbines with zero yaw
    # should be the same
    fmodel.reset_operation()
    fmodel.set(layout_x=[0, 200, 4000], layout_y=[0, 0, 0])
    fmodel.run_no_wake()
    power_no_wake = fmodel.get_turbine_powers()
    assert len(np.unique(power_no_wake)) == 1

def test_get_turbine_powers():
    # Get turbine powers should return n_findex x n_turbine powers
    # Apply the same wind speed and direction multiple times and confirm all equal

    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    n_turbines = len(layout_x)

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fmodel.run()

    turbine_powers = fmodel.get_turbine_powers()

    assert turbine_powers.shape[0] == n_findex
    assert turbine_powers.shape[1] == n_turbines
    assert turbine_powers[0, 0] == turbine_powers[1, 0]

def test_get_farm_power():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    # n_turbines = len(layout_x)

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fmodel.run()

    turbine_powers = fmodel.get_turbine_powers()
    farm_powers = fmodel.get_farm_power()

    assert farm_powers.shape[0] == n_findex

    # Assert farm power is the same as summing turbine powers
    # over the turbine axis
    farm_power_from_turbine = turbine_powers.sum(axis=1)
    np.testing.assert_almost_equal(farm_power_from_turbine, farm_powers)

    # Test using weights to disable the second turbine
    turbine_weights = np.array([1.0, 0.0])
    farm_powers = fmodel.get_farm_power(turbine_weights=turbine_weights)

    # Assert farm power is now equal to the 0th turbine since 1st is
    # disabled
    farm_power_from_turbine = turbine_powers[:, 0]
    np.testing.assert_almost_equal(farm_power_from_turbine, farm_powers)

    # Finally, test using weights only disable the 1 turbine on the final
    # findex values
    turbine_weights = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    farm_powers = fmodel.get_farm_power(turbine_weights=turbine_weights)
    turbine_powers[-1, 1] = 0
    farm_power_from_turbine = turbine_powers.sum(axis=1)
    np.testing.assert_almost_equal(farm_power_from_turbine, farm_powers)

def test_disable_turbines():

    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set to mixed turbine model
    with open(
        str(
            fmodel.core.as_dict()["farm"]["turbine_library_path"]
            / (fmodel.core.as_dict()["farm"]["turbine_type"][0] + ".yaml")
        )
    ) as t:
        turbine_type = yaml.safe_load(t)
    turbine_type["power_thrust_model"] = "mixed"
    fmodel.set(turbine_type=[turbine_type])

    # Init to n-findex = 2, n_turbines = 3
    fmodel.set(
        wind_speeds=np.array([8.,8.,]),
        wind_directions=np.array([270.,270.]),
        turbulence_intensities=np.array([0.06,0.06]),
        layout_x = [0,1000,2000],
        layout_y=[0,0,0]
    )

    # Confirm that using a disable value with wrong n_findex raises error
    with pytest.raises(ValueError):
        fmodel.set(disable_turbines=np.zeros((10, 3), dtype=bool))
        fmodel.run()

    # Confirm that using a disable value with wrong n_turbines raises error
    with pytest.raises(ValueError):
        fmodel.set(disable_turbines=np.zeros((2, 10), dtype=bool))
        fmodel.run()

    # Confirm that if all turbines are disabled, power is near 0 for all turbines
    fmodel.set(disable_turbines=np.ones((2, 3), dtype=bool))
    fmodel.run()
    turbines_powers = fmodel.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers, 0, atol=0.1)

    # Confirm the same for run_no_wake
    fmodel.run_no_wake()
    turbines_powers = fmodel.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers, 0, atol=0.1)

    # Confirm that if all disabled values set to false, equivalent to running normally
    fmodel.reset_operation()
    fmodel.run()
    turbines_powers_normal = fmodel.get_turbine_powers()
    fmodel.set(disable_turbines=np.zeros((2, 3), dtype=bool))
    fmodel.run()
    turbines_powers_false_disable = fmodel.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers_normal,turbines_powers_false_disable,atol=0.1)

    # Confirm the same for run_no_wake
    fmodel.run_no_wake()
    turbines_powers_normal = fmodel.get_turbine_powers()
    fmodel.set(disable_turbines=np.zeros((2, 3), dtype=bool))
    fmodel.run_no_wake()
    turbines_powers_false_disable = fmodel.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers_normal,turbines_powers_false_disable,atol=0.1)

    # Confirm the shutting off the middle turbine is like removing from the layout
    # In terms of impact on third turbine
    disable_turbines = np.zeros((2, 3), dtype=bool)
    disable_turbines[:,1] = [True, True]
    fmodel.set(disable_turbines=disable_turbines)
    fmodel.run()
    power_with_middle_disabled = fmodel.get_turbine_powers()

    # Two turbine case to compare against above
    fmodel_remove_middle = fmodel.copy()
    fmodel_remove_middle.set(layout_x=[0,2000], layout_y=[0, 0])
    fmodel_remove_middle.run()
    power_with_middle_removed = fmodel_remove_middle.get_turbine_powers()

    np.testing.assert_almost_equal(power_with_middle_disabled[0,2], power_with_middle_removed[0,1])
    np.testing.assert_almost_equal(power_with_middle_disabled[1,2], power_with_middle_removed[1,1])

    # Check that yaw angles are correctly set when turbines are disabled
    fmodel.set(
        layout_x=[0, 1000, 2000],
        layout_y=[0, 0, 0],
        disable_turbines=disable_turbines,
        yaw_angles=np.ones((2, 3))
    )
    fmodel.run()
    assert (fmodel.core.farm.yaw_angles == np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])).all()

def test_get_farm_aep():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    # n_turbines = len(layout_x)

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fmodel.run()

    farm_powers = fmodel.get_farm_power()

    # Start with uniform frequency
    freq = np.ones(n_findex)
    freq = freq / np.sum(freq)

    farm_aep = fmodel.get_farm_AEP(freq=freq)

    aep = np.sum(np.multiply(freq, farm_powers) * 365 * 24)

    # In this case farm_aep should match farm powers
    np.testing.assert_allclose(farm_aep, aep)

def test_get_farm_aep_with_conditions():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([5.0, 8.0, 8.0, 8.0, 20.0])
    wind_directions = np.array([270.0, 270.0, 270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06, 0.06, 0.06])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    # n_turbines = len(layout_x)

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fmodel.run()

    farm_powers = fmodel.get_farm_power()

    # Start with uniform frequency
    freq = np.ones(n_findex)
    freq = freq / np.sum(freq)

    # Get farm AEP with conditions on minimun and max wind speed
    # which exclude the first and last findex
    farm_aep = fmodel.get_farm_AEP(freq=freq, cut_in_wind_speed=6.0, cut_out_wind_speed=15.0)

    # In this case the aep should be computed assuming 0 power
    # for the 0th and last findex
    farm_powers[0] = 0
    farm_powers[-1] = 0
    aep = np.sum(np.multiply(freq, farm_powers) * 365 * 24)

    # In this case farm_aep should match farm powers
    np.testing.assert_allclose(farm_aep, aep)

    #Confirm n_findex reset after the operation
    assert n_findex == fmodel.core.flow_field.n_findex

def test_set_ti():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set wind directions, wind speeds and turbulence intensities with n_findex = 3
    fmodel.set(
        wind_speeds=[8.0, 8.0, 8.0],
        wind_directions=[240.0, 250.0, 260.0],
        turbulence_intensities=[0.1, 0.1, 0.1],
    )

    # Confirm can change turbulence intensities if not changing the length of the array
    fmodel.set(turbulence_intensities=[0.12, 0.12, 0.12])

    # Confirm that changes to wind speeds and directions without changing turbulence intensities
    # raises an error
    with pytest.raises(ValueError):
        fmodel.set(
            wind_speeds=[8.0, 8.0, 8.0, 8.0],
            wind_directions=[240.0, 250.0, 260.0, 270.0],
        )

    # Changing the length of TI alone is not allowed
    with pytest.raises(ValueError):
        fmodel.set(turbulence_intensities=[0.12])

    # Test that applying a float however raises an error
    with pytest.raises(TypeError):
        fmodel.set(turbulence_intensities=0.12)

def test_calculate_planes():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # The calculate_plane functions should run directly with the inputs as given
    fmodel.calculate_horizontal_plane(90.0)
    fmodel.calculate_y_plane(0.0)
    fmodel.calculate_cross_plane(500.0)

    # They should also support setting new wind conditions, but they all have to set at once
    wind_speeds = [8.0, 8.0, 8.0]
    wind_directions = [270.0, 270.0, 270.0]
    turbulence_intensities = [0.1, 0.1, 0.1]
    fmodel.calculate_horizontal_plane(
        90.0,
        ws=[wind_speeds[0]],
        wd=[wind_directions[0]],
        ti=[turbulence_intensities[0]]
    )
    fmodel.calculate_y_plane(
        0.0,
        ws=[wind_speeds[0]],
        wd=[wind_directions[0]],
        ti=[turbulence_intensities[0]]
    )
    fmodel.calculate_cross_plane(
        500.0,
        ws=[wind_speeds[0]],
        wd=[wind_directions[0]],
        ti=[turbulence_intensities[0]]
    )

    # If Floris is configured with multiple wind conditions prior to this, then all of the
    # components must be changed together.
    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities
    )
    with pytest.raises(ValueError):
        fmodel.calculate_horizontal_plane(90.0, ws=[wind_speeds[0]], wd=[wind_directions[0]])
    with pytest.raises(ValueError):
        fmodel.calculate_y_plane(0.0, ws=[wind_speeds[0]], wd=[wind_directions[0]])
    with pytest.raises(ValueError):
        fmodel.calculate_cross_plane(500.0, ws=[wind_speeds[0]], wd=[wind_directions[0]])

def test_get_and_set_param():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Get the wind speed
    wind_speeds = fmodel.get_param(['flow_field', 'wind_speeds'])
    assert wind_speeds[0] == 8.0

    # Set the wind speed
    fmodel.set_param(['flow_field', 'wind_speeds'], 10.0, param_idx=0)
    wind_speed = fmodel.get_param(['flow_field', 'wind_speeds'], param_idx=0  )
    assert wind_speed == 10.0

    # Repeat with wake parameter
    fmodel.set_param(['wake', 'wake_velocity_parameters', 'gauss', 'alpha'], 0.1)
    alpha = fmodel.get_param(['wake', 'wake_velocity_parameters', 'gauss', 'alpha'])
    assert alpha == 0.1

def test_get_power_thrust_model():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    assert fmodel.get_power_thrust_model() == "cosine-loss"

def test_set_power_thrust_model():

    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set_power_thrust_model("simple-derating")
    assert fmodel.get_power_thrust_model() == "simple-derating"
