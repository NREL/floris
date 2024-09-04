from pathlib import Path

import numpy as np
import pytest

from floris import FlorisModel
from floris.heterogeneous_map import HeterogeneousMap


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_declare_by_parameters():
    HeterogeneousMap(
        x=np.array([0.0, 0.0, 500.0, 500.0]),
        y=np.array([0.0, 500.0, 0.0, 500.0]),
        speed_multipliers=np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.25, 1.0, 1.25],
                [1.0, 1.0, 1.0, 1.25],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
        wind_directions=np.array([0.0, 0.0, 90.0, 90.0]),
        wind_speeds=np.array([5.0, 15.0, 5.0, 15.0]),
    )


def test_heterogeneous_map_no_ws_no_wd():
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
    }

    # Should be single value if no wind_directions or wind_speeds
    with pytest.raises(ValueError):
        HeterogeneousMap(**heterogeneous_map_config)

    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array([[1.0, 1.1, 1.2]]),
    }

    HeterogeneousMap(**heterogeneous_map_config)


def test_wind_direction_and_wind_speed_sizes():
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_directions": np.array([0.0, 90.0]),
        "wind_speeds": np.array([10.0, 20.0, 30.0]),
    }

    # Should raise value error because wind_directions and wind_speeds are not the same size
    with pytest.raises(ValueError):
        HeterogeneousMap(**heterogeneous_map_config)

        heterogeneous_map_config = {
            "x": np.array([0.0, 1.0, 2.0]),
            "y": np.array([0.0, 1.0, 2.0]),
            "speed_multipliers": np.array(
                [
                    [1.0, 1.1, 1.2],
                    [1.1, 1.1, 1.1],
                    [1.3, 1.4, 1.5],
                ]
            ),
            "wind_directions": np.array([0.0, 90.0]),
            "wind_speeds": np.array([10.0, 20.0]),
        }

    # Should raise value error because wind_directions and wind_speeds are not = to the
    # size of speed_multipliers in the 0th dimension
    with pytest.raises(ValueError):
        HeterogeneousMap(**heterogeneous_map_config)

    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_directions": np.array([0.0, 90.0, 270.0]),
        "wind_speeds": np.array([10.0, 20.0, 15.0]),
    }

    HeterogeneousMap(**heterogeneous_map_config)


def test_wind_direction_and_wind_speed_unique():
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_directions": np.array([0.0, 0.0, 270.0]),
        "wind_speeds": np.array([10.0, 10.0, 15.0]),
    }

    # Raises error because of repeated wd/ws pair
    with pytest.raises(ValueError):
        HeterogeneousMap(**heterogeneous_map_config)

    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_directions": np.array([0.0, 5.0, 270.0]),
        "wind_speeds": np.array([10.0, 10.0, 15.0]),
    }

    # Should not raise error
    HeterogeneousMap(**heterogeneous_map_config)


def test_get_heterogeneous_inflow_config_by_wind_direction():
    # Test the function when only wind_directions is defined
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_directions": np.array([0, 90, 270]),
    }

    # Check for correctness
    wind_directions = np.array([240, 80, 15])
    wind_speeds = np.array([10.0, 20.0, 15.0])
    expected_output = np.array([[1.3, 1.4, 1.5], [1.1, 1.1, 1.1], [1.0, 1.1, 1.2]])

    hm = HeterogeneousMap(**heterogeneous_map_config)
    output_dict = hm.get_heterogeneous_inflow_config(wind_directions, wind_speeds)
    assert np.allclose(output_dict["speed_multipliers"], expected_output)


def test_get_heterogeneous_inflow_config_by_wind_speed():
    # Test the function when only wind_directions is defined
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_speeds": np.array([0, 10, 20]),
    }

    # Check for correctness
    wind_directions = np.array([240, 80, 15])
    wind_speeds = np.array([10.0, 10.0, 18.0])
    expected_output = np.array([[1.1, 1.1, 1.1], [1.1, 1.1, 1.1], [1.3, 1.4, 1.5]])

    hm = HeterogeneousMap(**heterogeneous_map_config)
    output_dict = hm.get_heterogeneous_inflow_config(wind_directions, wind_speeds)
    assert np.allclose(output_dict["speed_multipliers"], expected_output)


def test_get_heterogeneous_inflow_config_by_wind_direction_and_wind_speed():
    # Test the function when only wind_directions is defined
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [[1.0, 1.1, 1.2], [1.1, 1.1, 1.1], [1.3, 1.4, 1.5], [1.4, 1.5, 1.6]]
        ),
        "wind_directions": np.array([0, 0, 90, 90]),
        "wind_speeds": np.array([5.0, 15.0, 5.0, 15.0]),
    }

    hm = HeterogeneousMap(**heterogeneous_map_config)

    # Check for correctness
    wind_directions = np.array([91, 89, 350])
    wind_speeds = np.array([4.0, 18.0, 12.0])
    expected_output = np.array([[1.3, 1.4, 1.5], [1.4, 1.5, 1.6], [1.1, 1.1, 1.1]])

    output_dict = hm.get_heterogeneous_inflow_config(wind_directions, wind_speeds)
    assert np.allclose(output_dict["speed_multipliers"], expected_output)


def test_get_heterogeneous_inflow_config_no_wind_direction_no_wind_speed():
    # Test the function when only wind_directions is defined
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
            ]
        ),
    }

    hm = HeterogeneousMap(**heterogeneous_map_config)

    # Check for correctness
    wind_directions = np.array([91, 89, 350])
    wind_speeds = np.array([4.0, 18.0, 12.0])
    expected_output = np.array([[1.0, 1.1, 1.2], [1.0, 1.1, 1.2], [1.0, 1.1, 1.2]])

    output_dict = hm.get_heterogeneous_inflow_config(wind_directions, wind_speeds)
    assert np.allclose(output_dict["speed_multipliers"], expected_output)


def test_get_2d_heterogenous_map_from_3d():
    hm_3d = HeterogeneousMap(
        x=np.array(
            [
                0.0,
                1.0,
                2.0,
                0.0,
                1.0,
                2.0,
            ]
        ),
        y=np.array(
            [
                0.0,
                1.0,
                2.0,
                0.0,
                1.0,
                2.0,
            ]
        ),
        z=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        speed_multipliers=np.array(
            [
                [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
                [1.1, 1.1, 1.1, 2.1, 2.1, 2.1],
                [1.3, 1.4, 1.5, 2.3, 2.4, 2.5],
            ]
        ),
        wind_directions=np.array([0, 90, 270]),
    )

    hm_2d_0 = hm_3d.get_heterogeneous_map_2d(0.0)
    hm_2d_1 = hm_3d.get_heterogeneous_map_2d(1.0)

    # Test that x values in both cases are 0, 1, 2
    assert np.allclose(hm_2d_0.x, np.array([0.0, 1.0, 2.0]))
    assert np.allclose(hm_2d_1.x, np.array([0.0, 1.0, 2.0]))

    # Test that the speed multipliers are correct
    assert np.allclose(
        hm_2d_0.speed_multipliers, np.array([[1.0, 1.1, 1.2], [1.1, 1.1, 1.1], [1.3, 1.4, 1.5]])
    )
    assert np.allclose(
        hm_2d_1.speed_multipliers, np.array([[2.0, 2.1, 2.2], [2.1, 2.1, 2.1], [2.3, 2.4, 2.5]])
    )

    # Test that wind directions are the same between 2d and 3d
    assert np.allclose(hm_2d_0.wind_directions, hm_3d.wind_directions)

    # Test that wind speed is None in all cases
    assert hm_3d.wind_speeds is None
    assert hm_2d_0.wind_speeds is None
    assert hm_2d_1.wind_speeds is None


def test_3d_het_and_shear():
    # Define a 3D het map with 4 z locations and a single x and y where
    # the speed ups are defined by the usual power low
    wind_speed = 8.0
    z_values_array = np.array([0.0, 100.0, 200.0, 300.0])
    reference_wind_height = 90.0
    wind_shear = 0.12
    speed_multipliers_array = (z_values_array / reference_wind_height) ** wind_shear

    # Define the x and y locations to be corners of a square that repeats
    # for 4 different z locations
    x_values = np.tile(np.array([-500.0, 500.0, -500.0, 500.0]), 4)
    y_values = np.tile(np.array([-500.0, -500.0, 500.0, 500.0]), 4)

    # Repeat each of the elements of z_values 4 times
    z_values = np.repeat(z_values_array, 4)
    speed_multipliers = np.repeat(speed_multipliers_array, 4)

    # Reshape speed_multipliers to be (1,16)
    speed_multipliers = speed_multipliers.reshape(1, -1)

    # Build the 3d HeterogeneousMap
    hm_3d = HeterogeneousMap(
        x=x_values,
        y=y_values,
        z=z_values,
        speed_multipliers=speed_multipliers,
    )

    # Get the FLORIS model
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set the model to a single wind direction, speed and turbine
    fmodel.set(
        wind_directions=[270.0],
        wind_speeds=[wind_speed],
        layout_x=[0],
        layout_y=[0],
        wind_shear=wind_shear,
    )

    # Run the model
    fmodel.run()

    # Get the velocities 100 m in front of the turbine
    # Use the calculate_y_plane method because sample_flow_at_points does not currently work
    # with heterogeneous inflow
    # u_at_points_shear = fmodel.sample_flow_at_points(x= -100.0 * np.ones_like(z_values_array),
    #                                               y= 0.0 * np.ones_like(z_values_array),
    #                                               z=z_values_array)
    y_plane = fmodel.calculate_y_plane(
        x_resolution=1,
        z_resolution=4,
        crossstream_dist=0.0,
        x_bounds=[-100.0, -100.0],
        z_bounds=[0.0, 300.0],
    )
    u_at_points_shear = y_plane.df["u"].values

    # Check that the velocities are as expected, ie the shear exponent of 0.12
    # produces speeds expeted from the power law
    assert np.allclose(u_at_points_shear, wind_speed * speed_multipliers_array)

    # Now change the model such that shear is 0, while the 3D het map is applied
    fmodel.set(
        wind_shear=0.0,
        heterogeneous_inflow_config=hm_3d.get_heterogeneous_inflow_config(
            wind_directions=[270.0], wind_speeds=[wind_speed]
        ),
    )

    # Run the model
    fmodel.run()

    # # Get the velocities 100 m in front of the turbine
    y_plane_het = fmodel.calculate_y_plane(
        x_resolution=1,
        z_resolution=4,
        crossstream_dist=0.0,
        x_bounds=[-100.0, -100.0],
        z_bounds=[0.0, 300.0],
    )
    u_at_points_het = y_plane_het.df["u"].values

    # Confirm this produces the same results as the shear model
    assert np.allclose(u_at_points_het, u_at_points_shear)

    # Finally confirm that applying both shear and het raises a value error
    with pytest.raises(ValueError):
        fmodel.set(
            wind_shear=0.12,
            heterogeneous_inflow_config=hm_3d.get_heterogeneous_inflow_config(
                wind_directions=[270.0], wind_speeds=[wind_speed]
            ),
        )
