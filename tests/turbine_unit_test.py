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

from floris.simulation import (
    average_velocity,
    axial_induction,
    power,
    thrust_coefficient,
    Turbine,
)
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
    assert turbine.power_thrust_table["pP"] == turbine_data["power_thrust_table"]["pP"]
    assert turbine.power_thrust_table["pT"] == turbine_data["power_thrust_table"]["pT"]
    assert turbine.TSR == turbine_data["TSR"]
    assert turbine.generator_efficiency == turbine_data["generator_efficiency"]
    assert (
        turbine.power_thrust_table["ref_air_density"]
        == turbine_data["power_thrust_table"]["ref_air_density"]
    )
    assert turbine.power_thrust_table["ref_tilt"] == turbine_data["power_thrust_table"]["ref_tilt"]
    assert np.array_equal(
        turbine.power_thrust_table["wind_speed"],
        turbine_data["power_thrust_table"]["wind_speed"]
    )
    assert np.array_equal(
        turbine.power_thrust_table["power"],
        turbine_data["power_thrust_table"]["power"]
    )
    assert np.array_equal(
        turbine.power_thrust_table["thrust_coefficient"],
        turbine_data["power_thrust_table"]["thrust_coefficient"]
    )
    assert turbine.rotor_radius == turbine.rotor_diameter / 2.0
    assert turbine.rotor_area == np.pi * turbine.rotor_radius ** 2.0

    # TODO: test these explicitly.
    # Test create a simpler interpolator and test that you get the values you expect
    # fCt_interp: interp1d = field(init=False)
    # power_function: interp1d = field(init=False)
    # tilt_interp: interp1d = field(init=False, default=None)

    assert callable(turbine.thrust_coefficient_function)
    assert callable(turbine.power_function)


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
    thrust = thrust_coefficient(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        yaw_angles=np.zeros((1, 1)),
        tilt_angles=np.ones((1, 1)) * 5.0,
        thrust_coefficient_functions={turbine.turbine_type: turbine.thrust_coefficient_function},
        tilt_interps={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False]]),
        turbine_type_map=turbine_type_map[:,0],
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
    )

    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    np.testing.assert_allclose(
        thrust,
        turbine_data["power_thrust_table"]["thrust_coefficient"][truth_index]
    )

    # Multiple turbines with index filter
    # 4 turbines with 3 x 3 grid arrays
    thrusts = thrust_coefficient(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 12 x 4 x 3 x 3
        yaw_angles=np.zeros((1, N_TURBINES)),
        tilt_angles=np.ones((1, N_TURBINES)) * 5.0,
        thrust_coefficient_functions={turbine.turbine_type: turbine.thrust_coefficient_function},
        tilt_interps={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False] * N_TURBINES]),
        turbine_type_map=turbine_type_map,
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
        ix_filter=INDEX_FILTER,
    )
    assert len(thrusts[0]) == len(INDEX_FILTER)

    for i in range(len(INDEX_FILTER)):
        truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(WIND_SPEEDS[0])
        np.testing.assert_allclose(
            thrusts[0, i],
            turbine_data["power_thrust_table"]["thrust_coefficient"][truth_index]
        )

    # Single floating turbine; note that 'tilt_interp' is not set to None
    thrust = thrust_coefficient(
        velocities=wind_speed * np.ones((1, 1, 3, 3)), # One findex, one turbine
        yaw_angles=np.zeros((1, 1)),
        tilt_angles=np.ones((1, 1)) * 5.0,
        thrust_coefficient_functions={
            turbine.turbine_type: turbine_floating.thrust_coefficient_function
        },
        tilt_interps={turbine_floating.turbine_type: turbine_floating.tilt_interp},
        correct_cp_ct_for_tilt=np.array([[True]]),
        turbine_type_map=turbine_type_map[:,0],
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
    )

    truth_index = turbine_floating_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    np.testing.assert_allclose(
        thrust,
        turbine_floating_data["power_thrust_table"]["thrust_coefficient"][truth_index]
    )


def test_power():
    # AIR_DENSITY = 1.225

    # Test that power is computed as expected for a single turbine
    n_turbines = 1
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(n_turbines * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    test_power = power(
        velocities=wind_speed * np.ones((1, 1, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine.power_thrust_table["ref_air_density"],
        power_functions={turbine.turbine_type: turbine.power_function},
        yaw_angles=np.zeros((1, 1)), # 1 findex, 1 turbine
        tilt_angles=turbine.power_thrust_table["ref_tilt"] * np.ones((1, 1)),
        tilt_interps={turbine.turbine_type: turbine.tilt_interp},
        turbine_type_map=turbine_type_map[:,0],
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
    )

    # Recompute using the provided power
    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    baseline_power = turbine_data["power_thrust_table"]["power"][truth_index] * 1000
    assert np.allclose(baseline_power, test_power)


    # At rated, the power calculated should be 5MW since the test data is the NREL 5MW turbine
    wind_speed = 18.0
    rated_power = power(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        air_density=turbine.power_thrust_table["ref_air_density"],
        power_functions={turbine.turbine_type: turbine.power_function},
        yaw_angles=np.zeros((1, 1)), # 1 findex, 1 turbine
        tilt_angles=turbine.power_thrust_table["ref_tilt"] * np.ones((1, 1)),
        tilt_interps={turbine.turbine_type: turbine.tilt_interp},
        turbine_type_map=turbine_type_map[:,0],
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
    )
    assert np.allclose(rated_power, 5e6)


    # At wind speed = 0.0, the power should be 0 based on the provided Cp curve
    wind_speed = 0.0
    zero_power = power(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        air_density=turbine.power_thrust_table["ref_air_density"],
        power_functions={turbine.turbine_type: turbine.power_function},
        yaw_angles=np.zeros((1, 1)), # 1 findex, 1 turbine
        tilt_angles=turbine.power_thrust_table["ref_tilt"] * np.ones((1, 1)),
        tilt_interps={turbine.turbine_type: turbine.tilt_interp},
        turbine_type_map=turbine_type_map[:,0],
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
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
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)),
        air_density=turbine.power_thrust_table["ref_air_density"],
        power_functions={turbine.turbine_type: turbine.power_function},
        yaw_angles=np.zeros((1, n_turbines)),
        tilt_angles=turbine.power_thrust_table["ref_tilt"] * np.ones((1, n_turbines)),
        tilt_interps={turbine.turbine_type: turbine.tilt_interp},
        turbine_type_map=turbine_type_map,
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
    )
    baseline_4_power = baseline_power * np.ones((1, n_turbines))
    assert np.allclose(baseline_4_power, test_4_power)
    assert np.shape(baseline_4_power) == np.shape(test_4_power)


    # Same as above but with the grid collapsed in the velocities array
    turbine_data = SampleInputs().turbine
    turbine = Turbine.from_dict(turbine_data)
    turbine_type_map = np.array(n_turbines * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]
    test_grid_power = power(
        velocities=wind_speed * np.ones((1, n_turbines, 1)),
        air_density=turbine.power_thrust_table["ref_air_density"],
        power_functions={turbine.turbine_type: turbine.power_function},
        yaw_angles=np.zeros((1, n_turbines)),
        tilt_angles=turbine.power_thrust_table["ref_tilt"] * np.ones((1, n_turbines)),
        tilt_interps={turbine.turbine_type: turbine.tilt_interp},
        turbine_type_map=turbine_type_map,
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
    )
    baseline_grid_power = baseline_power * np.ones((1, n_turbines))
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

    baseline_ai = 0.26752001107622186415

    # Single turbine
    wind_speed = 10.0
    ai = axial_induction(
        velocities=wind_speed * np.ones((1, 1, 3, 3)), # 1 findex, 1 Turbine
        yaw_angles=np.zeros((1, 1)),
        tilt_angles=np.ones((1, 1)) * 5.0,
        axial_induction_functions={turbine.turbine_type: turbine.axial_induction_function},
        tilt_interps={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False]]),
        turbine_type_map=turbine_type_map[0,0],
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
    )
    np.testing.assert_allclose(ai, baseline_ai)

    # Multiple turbines with ix filter
    ai = axial_induction(
        velocities=np.ones((N_TURBINES, 3, 3)) * WIND_CONDITION_BROADCAST,  # 12 x 4 x 3 x 3
        yaw_angles=np.zeros((1, N_TURBINES)),
        tilt_angles=np.ones((1, N_TURBINES)) * 5.0,
        axial_induction_functions={turbine.turbine_type: turbine.axial_induction_function},
        tilt_interps={turbine.turbine_type: None},
        correct_cp_ct_for_tilt=np.array([[False] * N_TURBINES]),
        turbine_type_map=turbine_type_map,
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
        ix_filter=INDEX_FILTER,
    )

    assert len(ai[0]) == len(INDEX_FILTER)

    # Test the 10 m/s wind speed to use the same baseline as above
    np.testing.assert_allclose(ai[2], baseline_ai)

    # Single floating turbine; note that 'tilt_interp' is not set to None
    ai = axial_induction(
        velocities=wind_speed * np.ones((1, 1, 3, 3)),
        yaw_angles=np.zeros((1, 1)),
        tilt_angles=np.ones((1, 1)) * 5.0,
        axial_induction_functions={turbine.turbine_type: turbine.axial_induction_function},
        tilt_interps={turbine_floating.turbine_type: turbine_floating.tilt_interp},
        correct_cp_ct_for_tilt=np.array([[True]]),
        turbine_type_map=turbine_type_map[0,0],
        turbine_power_thrust_tables={turbine.turbine_type: turbine.power_thrust_table},
    )
    np.testing.assert_allclose(ai, baseline_ai)


def test_asdict(sample_inputs_fixture: SampleInputs):

    turbine = Turbine.from_dict(sample_inputs_fixture.turbine)
    dict1 = turbine.as_dict()

    new_turb = Turbine.from_dict(dict1)
    dict2 = new_turb.as_dict()

    assert dict1 == dict2
