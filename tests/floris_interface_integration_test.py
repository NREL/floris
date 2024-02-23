from pathlib import Path

import numpy as np
import pytest
import yaml

from floris.simulation.turbine.operation_models import POWER_SETPOINT_DEFAULT
from floris.tools.floris_interface import FlorisInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_read_yaml():
    fi = FlorisInterface(configuration=YAML_INPUT)
    assert isinstance(fi, FlorisInterface)

def test_set_run():
    """
    These tests are designed to test the set / run sequence to ensure that inputs are
    set when they should be, not set when they shouldn't be, and that the run sequence
    retains or resets information as intended.
    """

    # In FLORIS v3.2, running calculate_wake twice incorrectly set the yaw angles when the first time
    # has non-zero yaw settings but the second run had all-zero yaw settings. The test below asserts
    # that the yaw angles are correctly set in subsequent calls to run.
    fi = FlorisInterface(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.set(yaw_angles=yaw_angles)
    fi.run()
    assert fi.floris.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.set(yaw_angles=yaw_angles)
    fi.run()
    assert fi.floris.farm.yaw_angles == yaw_angles

    # Verify making changes to the layout, wind speed, and wind direction both before and after
    # running the calculation
    fi.reset_operation()
    fi.set(layout_x=[0, 0], layout_y=[0, 1000], wind_speeds=[8, 8], wind_directions=[270, 270])
    assert np.array_equal(fi.floris.farm.layout_x, np.array([0, 0]))
    assert np.array_equal(fi.floris.farm.layout_y, np.array([0, 1000]))
    assert np.array_equal(fi.floris.flow_field.wind_speeds, np.array([8, 8]))
    assert np.array_equal(fi.floris.flow_field.wind_directions, np.array([270, 270]))

    # Double check that nothing has changed after running the calculation
    fi.run()
    assert np.array_equal(fi.floris.farm.layout_x, np.array([0, 0]))
    assert np.array_equal(fi.floris.farm.layout_y, np.array([0, 1000]))
    assert np.array_equal(fi.floris.flow_field.wind_speeds, np.array([8, 8]))
    assert np.array_equal(fi.floris.flow_field.wind_directions, np.array([270, 270]))

    # Verify that changing wind shear doesn't change the other settings above
    fi.set(wind_shear=0.1)
    assert fi.floris.flow_field.wind_shear == 0.1
    assert np.array_equal(fi.floris.farm.layout_x, np.array([0, 0]))
    assert np.array_equal(fi.floris.farm.layout_y, np.array([0, 1000]))
    assert np.array_equal(fi.floris.flow_field.wind_speeds, np.array([8, 8]))
    assert np.array_equal(fi.floris.flow_field.wind_directions, np.array([270, 270]))

    # Verify that operation set-points are retained after changing other settings
    yaw_angles = 20 * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.set(yaw_angles=yaw_angles)
    assert np.array_equal(fi.floris.farm.yaw_angles, yaw_angles)
    fi.set()
    assert np.array_equal(fi.floris.farm.yaw_angles, yaw_angles)
    fi.set(wind_speeds=[10, 10])
    assert np.array_equal(fi.floris.farm.yaw_angles, yaw_angles)
    power_setpoints = 1e6 * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.set(power_setpoints=power_setpoints)
    assert np.array_equal(fi.floris.farm.yaw_angles, yaw_angles)
    assert np.array_equal(fi.floris.farm.power_setpoints, power_setpoints)

    # Test that setting power setpoints through the .set() function actually sets the
    # power setpoints in the floris object
    fi.reset_operation()
    power_setpoints = 1e6 * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.set(power_setpoints=power_setpoints)
    fi.run()
    assert np.array_equal(fi.floris.farm.power_setpoints, power_setpoints)

    # Similar to above, any "None" set-points should be set to the default value
    power_setpoints = np.array([[1e6, None]])
    fi.set(layout_x=[0, 0], layout_y=[0, 1000], power_setpoints=power_setpoints)
    fi.run()
    assert np.array_equal(
        fi.floris.farm.power_setpoints,
        np.array([[power_setpoints[0, 0], POWER_SETPOINT_DEFAULT]])
    )

def test_reset_operation():
    # Calling the reset function should reset the power setpoints to the default values
    fi = FlorisInterface(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    power_setpoints = 1e6 * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.set(power_setpoints=power_setpoints, yaw_angles=yaw_angles)
    fi.run()
    fi.reset_operation()
    assert fi.floris.farm.yaw_angles == np.zeros(
        (fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines)
    )
    assert fi.floris.farm.power_setpoints == (
        POWER_SETPOINT_DEFAULT * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    )

    # Double check that running the calculate also doesn't change the operating set points
    fi.run()
    assert fi.floris.farm.yaw_angles == np.zeros(
        (fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines)
    )
    assert fi.floris.farm.power_setpoints == (
        POWER_SETPOINT_DEFAULT * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    )

def test_run_no_wake():
    # In FLORIS v3.2, running calculate_no_wake twice incorrectly set the yaw angles when the first
    # time has non-zero yaw settings but the second run had all-zero yaw settings. The test below
    # asserts that the yaw angles are correctly set in subsequent calls to run_no_wake.
    fi = FlorisInterface(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.set(yaw_angles=yaw_angles)
    fi.run_no_wake()
    assert fi.floris.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.set(yaw_angles=yaw_angles)
    fi.run_no_wake()
    assert fi.floris.farm.yaw_angles == yaw_angles

    # With no wake and three turbines in a line, the power for all turbines with zero yaw
    # should be the same
    fi.reset_operation()
    fi.set(layout_x=[0, 200, 4000], layout_y=[0, 0, 0])
    fi.run_no_wake()
    power_no_wake = fi.get_turbine_powers()
    assert len(np.unique(power_no_wake)) == 1

def test_get_turbine_powers():
    # Get turbine powers should return n_findex x n_turbine powers
    # Apply the same wind speed and direction multiple times and confirm all equal

    fi = FlorisInterface(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    n_turbines = len(layout_x)

    fi.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fi.run()

    turbine_powers = fi.get_turbine_powers()

    assert turbine_powers.shape[0] == n_findex
    assert turbine_powers.shape[1] == n_turbines
    assert turbine_powers[0, 0] == turbine_powers[1, 0]

def test_get_farm_power():
    fi = FlorisInterface(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    # n_turbines = len(layout_x)

    fi.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fi.run()

    turbine_powers = fi.get_turbine_powers()
    farm_powers = fi.get_farm_power()

    assert farm_powers.shape[0] == n_findex

    # Assert farm power is the same as summing turbine powers
    # over the turbine axis
    farm_power_from_turbine = turbine_powers.sum(axis=1)
    np.testing.assert_almost_equal(farm_power_from_turbine, farm_powers)

    # Test using weights to disable the second turbine
    turbine_weights = np.array([1.0, 0.0])
    farm_powers = fi.get_farm_power(turbine_weights=turbine_weights)

    # Assert farm power is now equal to the 0th turbine since 1st is
    # disabled
    farm_power_from_turbine = turbine_powers[:, 0]
    np.testing.assert_almost_equal(farm_power_from_turbine, farm_powers)

    # Finally, test using weights only disable the 1 turbine on the final
    # findex values
    turbine_weights = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    farm_powers = fi.get_farm_power(turbine_weights=turbine_weights)
    turbine_powers[-1, 1] = 0
    farm_power_from_turbine = turbine_powers.sum(axis=1)
    np.testing.assert_almost_equal(farm_power_from_turbine, farm_powers)

def test_disable_turbines():

    fi = FlorisInterface(configuration=YAML_INPUT)

    # Set to mixed turbine model
    with open(
        str(
            fi.floris.as_dict()["farm"]["turbine_library_path"]
            / (fi.floris.as_dict()["farm"]["turbine_type"][0] + ".yaml")
        )
    ) as t:
        turbine_type = yaml.safe_load(t)
    turbine_type["power_thrust_model"] = "mixed"
    fi.set(turbine_type=[turbine_type])

    # Init to n-findex = 2, n_turbines = 3
    fi.set(
        wind_speeds=np.array([8.,8.,]),
        wind_directions=np.array([270.,270.]),
        layout_x = [0,1000,2000],
        layout_y=[0,0,0]
    )

    # Confirm that passing in a disable value with wrong n_findex raises error
    with pytest.raises(ValueError):
        fi.set(disable_turbines=np.zeros((10, 3), dtype=bool))
        fi.run()

    # Confirm that passing in a disable value with wrong n_turbines raises error
    with pytest.raises(ValueError):
        fi.set(disable_turbines=np.zeros((2, 10), dtype=bool))
        fi.run()

    # Confirm that if all turbines are disabled, power is near 0 for all turbines
    fi.set(disable_turbines=np.ones((2, 3), dtype=bool))
    fi.run()
    turbines_powers = fi.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers,0,atol=0.1)

    # Confirm the same for run_no_wake
    fi.run_no_wake()
    turbines_powers = fi.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers,0,atol=0.1)

    # Confirm that if all disabled values set to false, equivalent to running normally
    fi.reset_operation()
    fi.run()
    turbines_powers_normal = fi.get_turbine_powers()
    fi.set(disable_turbines=np.zeros((2, 3), dtype=bool))
    fi.run()
    turbines_powers_false_disable = fi.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers_normal,turbines_powers_false_disable,atol=0.1)

    # Confirm the same for run_no_wake
    fi.run_no_wake()
    turbines_powers_normal = fi.get_turbine_powers()
    fi.set(disable_turbines=np.zeros((2, 3), dtype=bool))
    fi.run_no_wake()
    turbines_powers_false_disable = fi.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers_normal,turbines_powers_false_disable,atol=0.1)

    # Confirm the shutting off the middle turbine is like removing from the layout
    # In terms of impact on third turbine
    disable_turbines = np.zeros((2, 3), dtype=bool)
    disable_turbines[:,1] = [True, True]
    fi.set(disable_turbines=disable_turbines)
    fi.run()
    power_with_middle_disabled = fi.get_turbine_powers()

    fi.set(layout_x = [0,2000],layout_y = [0, 0], disable_turbines=np.zeros((2, 2), dtype=bool))
    fi.run()
    power_with_middle_removed = fi.get_turbine_powers()

    np.testing.assert_almost_equal(power_with_middle_disabled[0,2], power_with_middle_removed[0,1])
    np.testing.assert_almost_equal(power_with_middle_disabled[1,2], power_with_middle_removed[1,1])

    # Check that yaw angles are correctly set when turbines are disabled
    fi.set(
        layout_x=[0,1000,2000],
        layout_y=[0,0,0],
        disable_turbines=disable_turbines,
        yaw_angles=np.ones((2, 3))
    )
    fi.run()
    assert (fi.floris.farm.yaw_angles == np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])).all()

def test_get_farm_aep():
    fi = FlorisInterface(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    # n_turbines = len(layout_x)

    fi.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fi.run()

    farm_powers = fi.get_farm_power()

    # Start with uniform frequency
    freq = np.ones(n_findex)
    freq = freq / np.sum(freq)

    farm_aep = fi.get_farm_AEP(freq=freq)

    aep = np.sum(np.multiply(freq, farm_powers) * 365 * 24)

    # In this case farm_aep should match farm powers
    np.testing.assert_allclose(farm_aep, aep)

def test_get_farm_aep_with_conditions():
    fi = FlorisInterface(configuration=YAML_INPUT)

    wind_speeds = np.array([5.0, 8.0, 8.0, 8.0, 20.0])
    wind_directions = np.array([270.0, 270.0, 270.0, 270.0, 270.0])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    # n_turbines = len(layout_x)

    fi.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fi.run()

    farm_powers = fi.get_farm_power()

    # Start with uniform frequency
    freq = np.ones(n_findex)
    freq = freq / np.sum(freq)

    # Get farm AEP with conditions on minimun and max wind speed
    # which exclude the first and last findex
    farm_aep = fi.get_farm_AEP(freq=freq, cut_in_wind_speed=6.0, cut_out_wind_speed=15.0)

    # In this case the aep should be computed assuming 0 power
    # for the 0th and last findex
    farm_powers[0] = 0
    farm_powers[-1] = 0
    aep = np.sum(np.multiply(freq, farm_powers) * 365 * 24)

    # In this case farm_aep should match farm powers
    np.testing.assert_allclose(farm_aep, aep)

    #Confirm n_findex reset after the operation
    assert n_findex == fi.floris.flow_field.n_findex

def test_set_ti():
    fi = FlorisInterface(configuration=YAML_INPUT)

    # Set wind directions, wind speeds and turbulence intensities with n_findex = 3
    fi.set(
        wind_speeds=[8.0, 8.0, 8.0],
        wind_directions=[240.0, 250.0, 260.0],
        turbulence_intensities=[0.1, 0.1, 0.1],
    )

    # Now confirm can change wind speeds and directions shape without changing
    # turbulence intensity since this is allowed when the turbulence intensities are uniform
    # raises n_findex to 4
    fi.set(
        wind_speeds=[8.0, 8.0, 8.0, 8.0],
        wind_directions=[
            240.0,
            250.0,
            260.0,
            270.0,
        ],
    )

    # Confirm turbulence_intensities now length 4 with single unique value
    np.testing.assert_allclose(fi.floris.flow_field.turbulence_intensities, [0.1, 0.1, 0.1, 0.1])

    # Now should be able to change turbulence intensity to changing, so long as length 4
    fi.set(turbulence_intensities=[0.08, 0.09, 0.1, 0.11])

    # However the wrong length should raise an error
    with pytest.raises(ValueError):
        fi.set(turbulence_intensities=[0.08, 0.09, 0.1])

    # Also, now that TI is not a single unique value, it can not be left default when changing
    # shape of wind speeds and directions
    with pytest.raises(ValueError):
        fi.set(
            wind_speeds=[8.0, 8.0, 8.0, 8.0, 8.0],
            wind_directions=[
                240.0,
                250.0,
                260.0,
                270.0,
                280.0,
            ],
        )

    # Test that applying a 1D array of length 1 is allowed for ti
    fi.set(turbulence_intensities=[0.12])

    # Test that applying a float however raises an error
    with pytest.raises(TypeError):
        fi.set(turbulence_intensities=0.12)
