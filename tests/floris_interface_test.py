from pathlib import Path

import numpy as np

from floris.tools.floris_interface import FlorisInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full_v3.yaml"


def test_read_yaml():
    fi = FlorisInterface(configuration=YAML_INPUT)
    assert isinstance(fi, FlorisInterface)


def test_calculate_wake():
    """
    In FLORIS v3.2, running calculate_wake twice incorrectly set the yaw angles when the first time
    has non-zero yaw settings but the second run had all-zero yaw settings. The test below asserts
    that the yaw angles are correctly set in subsequent calls to calculate_wake.
    """
    fi = FlorisInterface(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.calculate_wake(yaw_angles=yaw_angles)
    assert fi.floris.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.calculate_wake(yaw_angles=yaw_angles)
    assert fi.floris.farm.yaw_angles == yaw_angles


def test_calculate_no_wake():
    """
    In FLORIS v3.2, running calculate_no_wake twice incorrectly set the yaw angles when the first
    time has non-zero yaw settings but the second run had all-zero yaw settings. The test below
    asserts that the yaw angles are correctly set in subsequent calls to calculate_no_wake.
    """
    fi = FlorisInterface(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.calculate_no_wake(yaw_angles=yaw_angles)
    assert fi.floris.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros((fi.floris.flow_field.n_findex, fi.floris.farm.n_turbines))
    fi.calculate_no_wake(yaw_angles=yaw_angles)
    assert fi.floris.farm.yaw_angles == yaw_angles


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

    fi.reinitialize(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fi.calculate_wake()

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

    fi.reinitialize(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fi.calculate_wake()

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


def test_get_farm_aep():
    fi = FlorisInterface(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    # n_turbines = len(layout_x)

    fi.reinitialize(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fi.calculate_wake()

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

    fi.reinitialize(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fi.calculate_wake()

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
