from pathlib import Path

import numpy as np
import pytest
import yaml

from floris import FlorisModel, TimeSeries
from floris.approx_floris_model import ApproxFlorisModel
from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT
from floris.uncertain_floris_model import UncertainFlorisModel, WindRose


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_read_yaml():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)
    assert isinstance(ufmodel, UncertainFlorisModel)


def test_rounded_inputs():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    # Using defaults
    # Example input array
    input_array = np.array([[45.3, 7.6, 0.24, 90.7, 749], [60.1, 8.2, 0.3, 95.3, 751]])

    # Expected output array after rounding
    expected_output = np.array([[45.0, 8.0, 0.25, 91.0, 700.0], [60.0, 8.0, 0.3, 95.0, 800.0]])

    # Call the function
    rounded_inputs = ufmodel._get_rounded_inputs(input_array)

    np.testing.assert_almost_equal(rounded_inputs, expected_output)


def test_expand_wind_directions():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    input_array = np.array(
        [[1, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120], [359, 140, 150]]
    )

    # Test even length
    with pytest.raises(ValueError):
        wd_sample_points = [-15, -10, -5, 5, 10, 15]  # Even lenght
        ufmodel._expand_wind_directions(input_array, wd_sample_points)

    # Test middle element not 0
    with pytest.raises(ValueError):
        wd_sample_points = [-15, -10, -5, 1, 5, 10, 15]  # Odd length, not 0 at the middle
        ufmodel._expand_wind_directions(input_array, wd_sample_points)

    # Test correction operations
    wd_sample_points = [-15, -10, -5, 0, 5, 10, 15]  # Odd length, 0 at the middle
    output_array = ufmodel._expand_wind_directions(input_array, wd_sample_points)

    # Check if output shape is correct
    assert output_array.shape[0] == 35

    # Check 360 wrapping
    # 1 - 15 = -14 -> 346
    np.testing.assert_almost_equal(output_array[0, 0], 346.0)

    # Check 360 wrapping
    # 359 + 15 = 374 -> 14
    np.testing.assert_almost_equal(output_array[-1, 0], 14.0)


def test_expand_wind_directions_with_yaw_nom():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    # Assume 2 turbine
    n_turbines = 2

    # Assume n_findex = 2
    input_array = np.array(
        [[270.0, 8.0, 0.6, 0.0, 0.0, 0.0, 0.0], [270.0, 8.0, 0.6, 0.0, 2.0, 0.0, 0.0]]
    )

    # 3 sample points
    wd_sample_points = [-3, 0, 3]

    # Test correction operations
    output_array = ufmodel._expand_wind_directions(input_array, wd_sample_points, True, n_turbines)

    # Check the first direction
    np.testing.assert_almost_equal(output_array[0, 0], 267)

    # Check the first yaw
    np.testing.assert_almost_equal(output_array[0, 4], -3)

    # Rerun with fix_yaw_to_nominal_direction = False, and now the yaw should be 0
    output_array = ufmodel._expand_wind_directions(input_array, wd_sample_points, False, n_turbines)

    # Check the first direction
    np.testing.assert_almost_equal(output_array[0, 0], 267)

    # Check the first yaw
    np.testing.assert_almost_equal(output_array[0, 4], 0)


def test_get_unique_inputs():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    input_array = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 1],
            [1, 1],
            [0, 1],
        ]
    )

    expected_unique_inputs = np.array([[0, 1], [0, 2], [1, 1]])

    unique_inputs, map_to_expanded_inputs = ufmodel._get_unique_inputs(input_array)

    # test expected result
    assert np.array_equal(unique_inputs, expected_unique_inputs)

    # Test gets back to original
    assert np.array_equal(unique_inputs[map_to_expanded_inputs], input_array)


def test_get_weights():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)
    weights = ufmodel._get_weights(3.0, [-6, -3, 0, 3, 6])
    np.testing.assert_allclose(
        weights, np.array([0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868])
    )


def test_uncertain_floris_model():
    # Recompute uncertain result using certain result with 1 deg

    fmodel = FlorisModel(configuration=YAML_INPUT)
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT, wd_sample_points=[-3, 0, 3], wd_std=3)

    fmodel.set(
        layout_x=[0, 300],
        layout_y=[0, 0],
        wind_speeds=[8.0, 8.0, 8.0],
        wind_directions=[267.0, 270.0, 273],
        turbulence_intensities=[0.06, 0.06, 0.06],
    )

    ufmodel.set(
        layout_x=[0, 300],
        layout_y=[0, 0],
        wind_speeds=[8.0],
        wind_directions=[270.0],
        turbulence_intensities=[0.06],
    )

    fmodel.run()
    ufmodel.run()

    nom_powers = fmodel.get_turbine_powers()[:, 1].flatten()
    unc_powers = ufmodel.get_turbine_powers()[:, 1].flatten()

    weights = ufmodel.weights

    np.testing.assert_allclose(np.sum(nom_powers * weights), unc_powers)


def test_uncertain_floris_model_setpoints():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT, wd_sample_points=[-3, 0, 3], wd_std=3)

    fmodel.set(
        layout_x=[0, 300],
        layout_y=[0, 0],
        wind_speeds=[8.0, 8.0, 8.0],
        wind_directions=[267.0, 270.0, 273],
        turbulence_intensities=[0.06, 0.06, 0.06],
    )

    ufmodel.set(
        layout_x=[0, 300],
        layout_y=[0, 0],
        wind_speeds=[8.0],
        wind_directions=[270.0],
        turbulence_intensities=[0.06],
    )
    weights = ufmodel.weights

    # Check setpoints dimensions are respected and reset_operation works
    # Note that fmodel.set() does NOT raise ValueError---an AttributeError is raised only at
    # fmodel.run()---whereas ufmodel.set raises ValueError immediately.
    # fmodel.set(yaw_angles=np.array([[0.0, 0.0]]))
    # with pytest.raises(AttributeError):
    #     fmodel.run()
    # with pytest.raises(ValueError):
    #     ufmodel.set(yaw_angles=np.array([[0.0, 0.0]]))

    fmodel.set(yaw_angles=np.array([[20.0, 0.0], [20.0, 0.0], [20.0, 0.0]]))
    fmodel.run()
    nom_powers = fmodel.get_turbine_powers()[:, 1].flatten()

    ufmodel.set(yaw_angles=np.array([[20.0, 0.0]]))
    ufmodel.run()
    unc_powers = ufmodel.get_turbine_powers()[:, 1].flatten()

    np.testing.assert_allclose(np.sum(nom_powers * weights), unc_powers)

    # Drop yaw setpoints and rerun
    fmodel.reset_operation()
    fmodel.run()
    nom_powers = fmodel.get_turbine_powers()[:, 1].flatten()

    ufmodel.reset_operation()
    ufmodel.run()
    unc_powers = ufmodel.get_turbine_powers()[:, 1].flatten()

    np.testing.assert_allclose(np.sum(nom_powers * weights), unc_powers)


def test_get_powers_with_wind_data():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 10.0, 12.0, 8.0, 10.0, 12.0])
    wind_directions = np.array([270.0, 270.0, 270.0, 280.0, 280.0, 280.0])
    turbulence_intensities = 0.06 * np.ones_like(wind_speeds)

    ufmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=[0, 1000, 2000, 3000],
        layout_y=[0, 0, 0, 0]
    )
    ufmodel.run()
    farm_power_simple = ufmodel.get_farm_power()

    # Now declare a WindRose with 2 wind directions and 3 wind speeds
    # uniform TI and frequency
    wind_rose = WindRose(
        wind_directions=np.unique(wind_directions),
        wind_speeds=np.unique(wind_speeds),
        ti_table=0.06
    )

    # Set this wind rose, run
    ufmodel.set(wind_data=wind_rose)
    ufmodel.run()

    farm_power_windrose = ufmodel.get_farm_power()

    # Check dimensions and that the farm power is the sum of the turbine powers
    assert farm_power_windrose.shape == (2, 3)
    assert np.allclose(farm_power_windrose, ufmodel.get_turbine_powers().sum(axis=2))

    # Check that simple and windrose powers are consistent
    assert np.allclose(farm_power_simple.reshape(2, 3), farm_power_windrose)
    assert np.allclose(farm_power_simple, farm_power_windrose.flatten())

    # Test that if the last turbine's weight is set to 0, the farm power is the same as the
    # sum of the first 3 turbines
    turbine_weights = np.array([1.0, 1.0, 1.0, 0.0])
    farm_power_weighted = ufmodel.get_farm_power(turbine_weights=turbine_weights)

    assert np.allclose(farm_power_weighted, ufmodel.get_turbine_powers()[:,:,:-1].sum(axis=2))

def test_approx_floris_model():

    afmodel = ApproxFlorisModel(configuration=YAML_INPUT, wd_resolution=1.0)

    time_series = TimeSeries(
        wind_directions = np.array([270.0, 270.1,271.0, 271.1]),
        wind_speeds=8.0,
        turbulence_intensities=0.06)

    afmodel.set(layout_x = np.array([0, 500]), layout_y = np.array([0, 0]), wind_data = time_series)

    # Test that 0th and 1th values are the same, as are the 2nd and 3rd
    afmodel.run()
    power = afmodel.get_farm_power()
    np.testing.assert_almost_equal(power[0], power[1])
    np.testing.assert_almost_equal(power[2], power[3])

    # Test with wind direction and wind speed varying
    afmodel = ApproxFlorisModel(configuration=YAML_INPUT, wd_resolution=1.0, ws_resolution=1.0)
    time_series = TimeSeries(
        wind_directions = np.array([270.0, 270.1,271.0, 271.1]),
        wind_speeds=np.array([8.0, 8.1, 8.0, 9.0]),
        turbulence_intensities=0.06)

    afmodel.set(layout_x = np.array([0, 500]), layout_y = np.array([0, 0]), wind_data = time_series)
    afmodel.run()

    # In this case the 0th and 1st should be the same, but not the 2nd and 3rd
    power = afmodel.get_farm_power()
    np.testing.assert_almost_equal(power[0], power[1])
    assert not np.allclose(power[2], power[3])
