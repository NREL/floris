import numpy as np
import pytest

from floris.heterogeneous_map import HeterogeneousMap


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
