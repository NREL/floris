from pathlib import Path

import numpy as np
import pytest
import yaml

from floris.simulation.turbine.operation_models import POWER_SETPOINT_DEFAULT
from floris.tools.floris_interface import FlorisInterface
from floris.tools.uncertainty_interface import UncertaintyInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_read_yaml():
    fi = UncertaintyInterface(configuration=YAML_INPUT)
    assert isinstance(fi, UncertaintyInterface)


def test_rounded_inputs():
    fi = UncertaintyInterface(configuration=YAML_INPUT)

    # Using defaults
    # Example input array
    input_array = np.array([[45.3, 7.6, 0.24, 90.7, 749], [60.1, 8.2, 0.3, 95.3, 751]])

    # Expected output array after rounding
    expected_output = np.array([[45.0, 8.0, 0.25, 91.0, 700.0], [60.0, 8.0, 0.3, 95.0, 800.0]])

    # Call the function
    rounded_inputs = fi._get_rounded_inputs(input_array)

    np.testing.assert_almost_equal(rounded_inputs, expected_output)


def test_expand_wind_directions():
    fi = UncertaintyInterface(configuration=YAML_INPUT)

    input_array = np.array(
        [[1, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120], [359, 140, 150]]
    )

    # Test even length
    with pytest.raises(ValueError):
        wd_sample_points = [-15, -10, -5, 5, 10, 15]  # Even lenght
        fi._expand_wind_directions(input_array, wd_sample_points)

    # Test middle element not 0
    with pytest.raises(ValueError):
        wd_sample_points = [-15, -10, -5, 1, 5, 10, 15]  # Odd length, not 0 at the middle
        fi._expand_wind_directions(input_array, wd_sample_points)

    # Test correction operations
    wd_sample_points = [-15, -10, -5, 0, 5, 10, 15]  # Odd length, 0 at the middle
    output_array = fi._expand_wind_directions(input_array, wd_sample_points)

    # Check if output shape is correct
    assert output_array.shape[0] == 35

    # Check 360 wrapping
    # 1 - 15 = -14 -> 346
    np.testing.assert_almost_equal(output_array[0, 0], 346.0)

    # Check 360 wrapping
    # 359 + 15 = 374 -> 14
    np.testing.assert_almost_equal(output_array[-1, 0], 14.0)


def test_get_unique_inputs():
    fi = UncertaintyInterface(configuration=YAML_INPUT)

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

    unique_inputs, map_to_expanded_inputs = fi._get_unique_inputs(input_array)

    # test expected result
    assert np.array_equal(unique_inputs, expected_unique_inputs)

    # Test gets back to original
    assert np.array_equal(unique_inputs[map_to_expanded_inputs], input_array)


def test_get_weights():
    fi = UncertaintyInterface(configuration=YAML_INPUT)
    weights = fi._get_weights(3.0, [-6, -3, 0, 3, 6])
    np.testing.assert_allclose(
        weights, np.array([0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868])
    )
