# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import os
from pathlib import Path

import attr
import numpy as np
import pytest
import yaml
from scipy.interpolate import interp1d

from floris.simulation import (
    average_velocity,
    axial_induction,
    Ct,
    power,
    Turbine,
)
from floris.simulation.turbine import (
    _rotor_velocity_tilt_correction,
    _rotor_velocity_yaw_correction,
    compute_tilt_angles_for_floating_turbines,
    cubic_cubature,
    simple_cubature,
)
from floris.turbine_library import build_turbine_dict
from tests.conftest import SampleInputs, WIND_SPEEDS


# size 16 x 1 x 1 x 1
# 16 wind speed and wind direction combinations from conftest
WIND_CONDITION_BROADCAST = np.reshape(np.array(WIND_SPEEDS), (-1, 1, 1, 1))

INDEX_FILTER = [0, 2]


def test_turbine_init():
    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)
    assert turbine.turbine_type == turbine_data["turbine_type"]
    assert turbine.rotor_diameter == turbine_data["rotor_diameter"]
    assert turbine.hub_height == turbine_data["hub_height"]
    assert turbine.pP == turbine_data["pP"]
    assert turbine.pT == turbine_data["pT"]
    assert turbine.TSR == turbine_data["TSR"]
    assert turbine.generator_efficiency == turbine_data["generator_efficiency"]
    assert turbine.ref_density_cp_ct == turbine_data["ref_density_cp_ct"]
    assert turbine.ref_tilt_cp_ct == turbine_data["ref_tilt_cp_ct"]
    assert np.array_equal(
        turbine.power_thrust_table["wind_speed"],
        turbine_data["power_thrust_table"]["wind_speed"]
    )
    assert np.array_equal(
        turbine.power_thrust_table["power"],
        turbine_data["power_thrust_table"]["power"]
    )
    assert np.array_equal(
        turbine.power_thrust_table["thrust"],
        turbine_data["power_thrust_table"]["thrust"]
    )
    assert turbine.rotor_radius == turbine.rotor_diameter / 2.0
    assert turbine.rotor_area == np.pi * turbine.rotor_radius ** 2.0

    # TODO: test these explicitly.
    # Test create a simpler interpolator and test that you get the values you expect
    # fCt_interp: interp1d = field(init=False)
    # power_interp: interp1d = field(init=False)
    # tilt_interp: interp1d = field(init=False, default=None)

    assert isinstance(turbine.fCt_interp, interp1d)
    assert isinstance(turbine.power_interp, interp1d)


def test_rotor_radius():

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)

    # Test that the radius is set correctly from the input file
    assert turbine.rotor_radius == turbine_data["rotor_diameter"] / 2.0

    # Test the radius setter method since it actually sets the diameter
    turbine.rotor_radius = 200.0
    assert turbine.rotor_diameter == 400.0

    # Test the getter-method again
    assert turbine.rotor_radius == 200.0


def test_rotor_area():

    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)

    # Test that the area is set correctly from the input file
    assert turbine.rotor_area == np.pi * (turbine_data["rotor_diameter"] / 2.0) ** 2

    # Test the area setter method since it actually sets the radius and then the diameter
    turbine.rotor_area = np.pi
    assert turbine.rotor_radius == 1
    assert turbine.rotor_diameter == 2

    # Test the getter-method again
    assert turbine.rotor_area == np.pi


def test_average_velocity():
    # TODO: why do we use cube root - mean - cube (like rms) instead of a simple average (np.mean)?
    # Dimensions are (n_findex, n turbines, grid x, grid y)
    velocities = np.ones((1, 1, 5, 5))
    assert average_velocity(velocities, method="cubic-mean") == 1

    # Constructs an array of shape 1 x 2 x 3 x 3 with first turbine all 1, second turbine all 2
    velocities = np.stack(
        (
            np.ones((1, 3, 3)),  # The first dimension here is the findex dimension and the second
            2 * np.ones((1, 3, 3)),  # is the n turbine since we are stacking on axis=1
        ),
        axis=1,
    )

    # Pull out the first findex for the test
    np.testing.assert_array_equal(
        average_velocity(velocities, method="cubic-mean")[0],
        np.array([1, 2])
    )

    # Test boolean filter
    ix_filter = [True, False, True, False]
    velocities = np.stack(  # 4 turbines with 3 x 3 velocity array; shape (1,4,3,3)
        [i * np.ones((1, 3, 3)) for i in range(1,5)],
        # (
        #     # The first dimension here is the findex dimension
        #     # and second is the turbine dimension since we are stacking on axis=1
        #     np.ones(
        #         (1, 3, 3)
        #     ),
        #     2 * np.ones((1, 3, 3)),
        #     3 * np.ones((1, 3, 3)),
        #     4 * np.ones((1, 3, 3)),
        # ),
        axis=1,
    )
    avg = average_velocity(velocities, ix_filter, method="cubic-mean")
    assert avg.shape == (1, 2)  # 1 = n_findex, 2 turbines filtered

    # Pull out the first findex for the comparison
    assert np.allclose(avg[0], np.array([1.0, 3.0]))
    # This fails in GitHub Actions due to a difference in precision:
    # E           assert 3.0000000000000004 == 3.0
    # np.testing.assert_array_equal(avg[0], np.array([1.0, 3.0]))

    # Test integer array filter
    # np.arange(1, 5).reshape((-1,1,1)) * np.ones((1, 1, 3, 3))
    velocities = np.stack(  # 4 turbines with 3 x 3 velocity array; shape (1,4,3,3)
        [i * np.ones((1, 3, 3)) for i in range(1,5)],
        axis=1,
    )
    avg = average_velocity(velocities, INDEX_FILTER, method="cubic-mean")
    assert avg.shape == (1, 2)  # 1 findex, 2 turbines filtered

    # Pull out the first findex for the comparison
    assert np.allclose(avg[0], np.array([1.0, 3.0]))


def test_ct():
    N_TURBINES = 4

    turbine_data = SampleInputs().turbine
    turbine_floating_data = SampleInputs().turbine_floating
    turbine = Turbine.from_dict(turbine_data)
    turbine_floating = Turbine.from_dict(turbine_floating_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])

    # Add the findex (0th) dimension
    turbine_type_map = turbine_type_map[None, :]

    # Single turbine
    # yaw angle / fCt are (n_findex, n turbine)
    wind_speed = 10.0
    thrust = Ct(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1)),
        tilt_angle=np.ones((1, 1)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1)) * 5.0,
        fCt={turbine.turbine_type: turbine.fCt_interp},
        tilt_interp={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False]]),
        turbine_type_map=turbine_type_map[:,0]
    )

    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    np.testing.assert_allclose(thrust, turbine_data["power_thrust_table"]["thrust"][truth_index])

    # Multiple turbines with index filter
    # 4 turbines with 3 x 3 grid arrays
    thrusts = Ct(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 12 x 4 x 3 x 3
        yaw_angle=np.zeros((1, N_TURBINES)),
        tilt_angle=np.ones((1, N_TURBINES)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, N_TURBINES)) * 5.0,
        fCt={turbine.turbine_type: turbine.fCt_interp},
        tilt_interp={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False] * N_TURBINES]),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
    )
    assert len(thrusts[0]) == len(INDEX_FILTER)

    for i in range(len(INDEX_FILTER)):
        truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(WIND_SPEEDS[0])
        np.testing.assert_allclose(
            thrusts[0, i],
            turbine_data["power_thrust_table"]["thrust"][truth_index]
        )

    # Single floating turbine; note that 'tilt_interp' is not set to None
    thrust = Ct(
        velocities=wind_speed * np.ones((1, 1, 3, 3)), # One findex, one turbine
        yaw_angle=np.zeros((1, 1)),
        tilt_angle=np.ones((1, 1)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1)) * 5.0,
        fCt={turbine.turbine_type: turbine_floating.fCt_interp},
        tilt_interp={turbine_floating.turbine_type: turbine_floating.tilt_interp},
        correct_cp_ct_for_tilt=np.array([[True]]),
        turbine_type_map=turbine_type_map[:,0]
    )

    truth_index = turbine_floating_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    np.testing.assert_allclose(
        thrust,
        turbine_floating_data["power_thrust_table"]["thrust"][truth_index]
    )


def test_power():
    AIR_DENSITY = 1.225

    # Test that power is computed as expected for a single turbine
    n_turbines = 1
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(n_turbines * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    test_power = power(
        ref_density_cp_ct=AIR_DENSITY,
        rotor_effective_velocities=wind_speed * np.ones((1, 1)), # 1 findex, 1 turbine
        power_interp={turbine.turbine_type: turbine.power_interp},
        turbine_type_map=turbine_type_map[:,0]
    )

    # Recompute using the provided Cp table
    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    cp_truth = turbine_data["power_thrust_table"]["power"][truth_index]
    baseline_power = (
        0.5
        * cp_truth
        * AIR_DENSITY
        * turbine.rotor_area
        * wind_speed ** 3
        * turbine.generator_efficiency
    )
    assert np.allclose(baseline_power, test_power)


    # At rated, the power calculated should be 5MW since the test data is the NREL 5MW turbine
    wind_speed = 18.0
    rated_power = power(
        ref_density_cp_ct=AIR_DENSITY,
        rotor_effective_velocities=wind_speed * np.ones((1, 1, 1)),
        power_interp={turbine.turbine_type: turbine.power_interp},
        turbine_type_map=turbine_type_map[:,0]
    )
    assert np.allclose(rated_power, 5e6)


    # At wind speed = 0.0, the power should be 0 based on the provided Cp curve
    wind_speed = 0.0
    zero_power = power(
        ref_density_cp_ct=AIR_DENSITY,
        rotor_effective_velocities=wind_speed * np.ones((1, 1, 1)),
        power_interp={turbine.turbine_type: turbine.power_interp},
        turbine_type_map=turbine_type_map[:,0]
    )
    assert np.allclose(zero_power, 0.0)


    # Test 4-turbine velocities array
    n_turbines = 4
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(n_turbines * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    test_4_power = power(
        ref_density_cp_ct=AIR_DENSITY,
        rotor_effective_velocities=wind_speed * np.ones((1, 1, n_turbines)),
        power_interp={turbine.turbine_type: turbine.power_interp},
        turbine_type_map=turbine_type_map
    )
    baseline_4_power = baseline_power * np.ones((1, 1, n_turbines))
    assert np.allclose(baseline_4_power, test_4_power)
    assert np.shape(baseline_4_power) == np.shape(test_4_power)


    # Same as above but with the grid expanded in the velocities array
    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(n_turbines * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    test_grid_power = power(
        ref_density_cp_ct=AIR_DENSITY,
        rotor_effective_velocities=wind_speed * np.ones((1, 1, n_turbines, 3, 3)),
        power_interp={turbine.turbine_type: turbine.power_interp},
        turbine_type_map=turbine_type_map[:,0]
    )
    baseline_grid_power = baseline_power * np.ones((1, 1, n_turbines, 3, 3))
    assert np.allclose(baseline_grid_power, test_grid_power)
    assert np.shape(baseline_grid_power) == np.shape(test_grid_power)


def test_axial_induction():

    N_TURBINES = 4

    turbine_data = SampleInputs().turbine
    turbine_floating_data = SampleInputs().turbine_floating
    turbine = Turbine.from_dict(turbine_data)
    turbine_floating = Turbine.from_dict(turbine_floating_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]

    baseline_ai = 0.25116283939089806

    # Single turbine
    wind_speed = 10.0
    ai = axial_induction(
        velocities=wind_speed * np.ones((1, 1, 3, 3)), # 1 findex, 1 Turbine
        yaw_angle=np.zeros((1, 1)),
        tilt_angle=np.ones((1, 1)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1)) * 5.0,
        fCt={turbine.turbine_type: turbine.fCt_interp},
        tilt_interp={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False]]),
        turbine_type_map=turbine_type_map[0,0],
    )
    np.testing.assert_allclose(ai, baseline_ai)

    # Multiple turbines with ix filter
    ai = axial_induction(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 12 x 4 x 3 x 3
        yaw_angle=np.zeros((1, N_TURBINES)),
        tilt_angle=np.ones((1, N_TURBINES)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, N_TURBINES)) * 5.0,
        fCt={turbine.turbine_type: turbine.fCt_interp},
        tilt_interp={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False] * N_TURBINES]),
        turbine_type_map=turbine_type_map,
        ix_filter=INDEX_FILTER,
    )

    assert len(ai[0]) == len(INDEX_FILTER)

    # Test the 10 m/s wind speed to use the same baseline as above
    np.testing.assert_allclose(ai[2], baseline_ai)

    # Single floating turbine; note that 'tilt_interp' is not set to None
    ai = axial_induction(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        yaw_angle=np.zeros((1, 1)),
        tilt_angle=np.ones((1, 1)) * 5.0,
        ref_tilt_cp_ct=np.ones((1, 1)) * 5.0,
        fCt={turbine.turbine_type: turbine_floating.fCt_interp},
        tilt_interp={turbine_floating.turbine_type: turbine_floating.tilt_interp},
        correct_cp_ct_for_tilt=np.array([[True]]),
        turbine_type_map=turbine_type_map[0,0],
    )
    np.testing.assert_allclose(ai, baseline_ai)


def test_rotor_velocity_yaw_correction():
    N_TURBINES = 4

    wind_speed = average_velocity(10.0 * np.ones((1, 1, 3, 3)))
    wind_speed_N_TURBINES = average_velocity(10.0 * np.ones((1, N_TURBINES, 3, 3)))

    # Test a single turbine for zero yaw
    yaw_corrected_velocities = _rotor_velocity_yaw_correction(
        pP=3.0,
        yaw_angle=0.0,
        rotor_effective_velocities=wind_speed,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, wind_speed)

    # Test a single turbine for non-zero yaw
    yaw_corrected_velocities = _rotor_velocity_yaw_correction(
        pP=3.0,
        yaw_angle=60.0,
        rotor_effective_velocities=wind_speed,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, 0.5 * wind_speed)

    # Test multiple turbines for zero yaw
    yaw_corrected_velocities = _rotor_velocity_yaw_correction(
        pP=3.0,
        yaw_angle=np.zeros((1, N_TURBINES)),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, wind_speed_N_TURBINES)

    # Test multiple turbines for non-zero yaw
    yaw_corrected_velocities = _rotor_velocity_yaw_correction(
        pP=3.0,
        yaw_angle=np.ones((1, N_TURBINES)) * 60.0,
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, 0.5 * wind_speed_N_TURBINES)


def test_rotor_velocity_tilt_correction():
    N_TURBINES = 4

    wind_speed = average_velocity(10.0 * np.ones((1, 1, 3, 3)))
    wind_speed_N_TURBINES = average_velocity(10.0 * np.ones((1, N_TURBINES, 3, 3)))

    turbine_data = SampleInputs().turbine
    turbine_floating_data = SampleInputs().turbine_floating
    turbine = Turbine.from_dict(turbine_data)
    turbine_floating = Turbine.from_dict(turbine_floating_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]

    # Test single non-floating turbine
    tilt_corrected_velocities = _rotor_velocity_tilt_correction(
        turbine_type_map=np.array([turbine_type_map[:, 0]]),
        tilt_angle=5.0*np.ones((1, 1)),
        ref_tilt_cp_ct=np.array([turbine.ref_tilt_cp_ct]),
        pT=np.array([turbine.pT]),
        tilt_interp={turbine.turbine_type: turbine.tilt_interp},
        correct_cp_ct_for_tilt=np.array([[False]]),
        rotor_effective_velocities=wind_speed,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed)

    # Test multiple non-floating turbines
    tilt_corrected_velocities = _rotor_velocity_tilt_correction(
        turbine_type_map=turbine_type_map,
        tilt_angle=5.0*np.ones((1, N_TURBINES)),
        ref_tilt_cp_ct=np.array([turbine.ref_tilt_cp_ct] * N_TURBINES),
        pT=np.array([turbine.pT] * N_TURBINES),
        tilt_interp={turbine.turbine_type: turbine.tilt_interp},
        correct_cp_ct_for_tilt=np.array([[False] * N_TURBINES]),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed_N_TURBINES)

    # Test single floating turbine
    tilt_corrected_velocities = _rotor_velocity_tilt_correction(
        turbine_type_map=np.array([turbine_type_map[:, 0]]),
        tilt_angle=5.0*np.ones((1, 1)),
        ref_tilt_cp_ct=np.array([turbine_floating.ref_tilt_cp_ct]),
        pT=np.array([turbine_floating.pT]),
        tilt_interp={turbine_floating.turbine_type: turbine_floating.tilt_interp},
        correct_cp_ct_for_tilt=np.array([[True]]),
        rotor_effective_velocities=wind_speed,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed)

    # Test multiple floating turbines
    tilt_corrected_velocities = _rotor_velocity_tilt_correction(
        turbine_type_map,
        tilt_angle=5.0*np.ones((1, N_TURBINES)),
        ref_tilt_cp_ct=np.array([turbine_floating.ref_tilt_cp_ct] * N_TURBINES),
        pT=np.array([turbine_floating.pT] * N_TURBINES),
        tilt_interp={turbine_floating.turbine_type: turbine_floating.tilt_interp},
        correct_cp_ct_for_tilt=np.array([[True] * N_TURBINES]),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed_N_TURBINES)


def test_compute_tilt_angles_for_floating_turbines():
    N_TURBINES = 4

    wind_speed = 25.0
    rotor_effective_velocities = average_velocity(wind_speed * np.ones((1, 1, 3, 3)))
    rotor_effective_velocities_N_TURBINES = average_velocity(
        wind_speed * np.ones((1, N_TURBINES, 3, 3))
    )

    turbine_floating_data = SampleInputs().turbine_floating
    turbine_floating = Turbine.from_dict(turbine_floating_data)
    turbine_type_map = np.array(N_TURBINES * [turbine_floating.turbine_type])
    turbine_type_map = turbine_type_map[None, :]

    # Single turbine
    tilt = compute_tilt_angles_for_floating_turbines(
        turbine_type_map=np.array([turbine_type_map[:, 0]]),
        tilt_angle=5.0*np.ones((1, 1)),
        tilt_interp={turbine_floating.turbine_type: turbine_floating.tilt_interp},
        rotor_effective_velocities=rotor_effective_velocities,
    )

    # calculate tilt again
    truth_index = turbine_floating_data["floating_tilt_table"]["wind_speed"].index(wind_speed)
    tilt_truth = turbine_floating_data["floating_tilt_table"]["tilt"][truth_index]
    np.testing.assert_allclose(tilt, tilt_truth)

    # Multiple turbines
    tilt_N_turbines = compute_tilt_angles_for_floating_turbines(
        turbine_type_map=np.array(turbine_type_map),
        tilt_angle=5.0*np.ones((1, N_TURBINES)),
        tilt_interp={turbine_floating.turbine_type: turbine_floating.tilt_interp},
        rotor_effective_velocities=rotor_effective_velocities_N_TURBINES,
    )

    # calculate tilt again
    truth_index = turbine_floating_data["floating_tilt_table"]["wind_speed"].index(wind_speed)
    tilt_truth = turbine_floating_data["floating_tilt_table"]["tilt"][truth_index]
    np.testing.assert_allclose(tilt_N_turbines, [[tilt_truth] * N_TURBINES])


def test_asdict(sample_inputs_fixture: SampleInputs):

    turbine = Turbine.from_dict(sample_inputs_fixture.turbine)
    dict1 = turbine.as_dict()

    new_turb = Turbine.from_dict(dict1)
    dict2 = new_turb.as_dict()

    assert dict1 == dict2


def test_simple_cubature():

    # Define a velocity array
    velocities = np.ones((1, 1, 3, 3))

    # Define sample cubature weights
    cubature_weights = np.array([1., 1., 1.])

    # Define the axis as last 2 dimensions
    axis = (velocities.ndim-2, velocities.ndim-1)

    # Calculate expected output based on the given inputs
    expected_output = 1.0

    # Call the function with the given inputs
    result = simple_cubature(velocities, cubature_weights, axis)

    # Check if the result matches the expected output
    np.testing.assert_allclose(result, expected_output)

def test_cubic_cubature():

    # Define a velocity array
    velocities = np.ones((1, 1, 3, 3))

    # Define sample cubature weights
    cubature_weights = np.array([1., 1., 1.])

    # Define the axis as last 2 dimensions
    axis = (velocities.ndim-2, velocities.ndim-1)

    # Calculate expected output based on the given inputs
    expected_output = 1.0

    # Call the function with the given inputs
    result = cubic_cubature(velocities, cubature_weights, axis)

    # Check if the result matches the expected output
    np.testing.assert_allclose(result, expected_output)

def test_build_turbine_dict():

    orig_file_path = Path(__file__).resolve().parent / "data" / "nrel_5MW_custom.yaml"
    test_turb_name = "test_turbine_export"
    test_file_path = "."

    in_dict = yaml.safe_load( open(orig_file_path, "r") )

    # Mocked up turbine data
    turbine_data_dict = {
        "wind_speed":in_dict["power_thrust_table"]["wind_speed"],
        "power_coefficient":in_dict["power_thrust_table"]["power"],
        "thrust_coefficient":in_dict["power_thrust_table"]["thrust"]
    }

    build_turbine_dict(
        turbine_data_dict,
        test_turb_name,
        file_path=test_file_path,
        generator_efficiency=in_dict["generator_efficiency"],
        hub_height=in_dict["hub_height"],
        pP=in_dict["pP"],
        pT=in_dict["pT"],
        rotor_diameter=in_dict["rotor_diameter"],
        TSR=in_dict["TSR"],
        air_density=in_dict["ref_density_cp_ct"],
        ref_tilt_cp_ct=in_dict["ref_tilt_cp_ct"]
    )

    test_dict = yaml.safe_load(
        open(os.path.join(test_file_path, test_turb_name+".yaml"), "r")
    )

    # Correct intended difference for test; assert equal
    test_dict["turbine_type"] = in_dict["turbine_type"]
    assert list(in_dict.keys()) == list(test_dict.keys())
    assert in_dict == test_dict

    # Now, in absolute values
    Cp = np.array(in_dict["power_thrust_table"]["power"])
    Ct = np.array(in_dict["power_thrust_table"]["thrust"])
    ws = np.array(in_dict["power_thrust_table"]["wind_speed"])

    P = 0.5 * in_dict["ref_density_cp_ct"] * (np.pi * in_dict["rotor_diameter"]**2/4) \
        * Cp * ws**3
    T = 0.5 * in_dict["ref_density_cp_ct"] * (np.pi * in_dict["rotor_diameter"]**2/4) \
        * Ct * ws**2

    turbine_data_dict = {
        "wind_speed":in_dict["power_thrust_table"]["wind_speed"],
        "power_absolute": P/1000,
        "thrust_absolute": T/1000
    }

    build_turbine_dict(
        turbine_data_dict,
        test_turb_name,
        file_path=test_file_path,
        generator_efficiency=in_dict["generator_efficiency"],
        hub_height=in_dict["hub_height"],
        pP=in_dict["pP"],
        pT=in_dict["pT"],
        rotor_diameter=in_dict["rotor_diameter"],
        TSR=in_dict["TSR"],
        air_density=in_dict["ref_density_cp_ct"],
        ref_tilt_cp_ct=in_dict["ref_tilt_cp_ct"]
    )

    test_dict = yaml.safe_load(
        open(os.path.join(test_file_path, test_turb_name+".yaml"), "r")
    )

    test_dict["turbine_type"] = in_dict["turbine_type"]
    assert list(in_dict.keys()) == list(test_dict.keys())
    for k in in_dict.keys():
        if type(in_dict[k]) is dict:
            for k2 in in_dict[k].keys():
                assert np.allclose(in_dict[k][k2], test_dict[k][k2])
        elif type(in_dict[k]) is str:
            assert in_dict[k] == test_dict[k]
        else:
            assert np.allclose(in_dict[k], test_dict[k])

    os.remove( os.path.join(test_file_path, test_turb_name+".yaml") )
